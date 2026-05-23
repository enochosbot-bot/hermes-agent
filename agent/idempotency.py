"""
idempotency.py — side-effect durability for the Hermes graph runtime.

Implements spec v0.3 §4.5:

  - `logical_action_id` is computed at the moment a side effect is first
    scheduled and persisted in state. Crucially, it does NOT include
    `run_id`. A replacement run for the same thread reuses the existing
    logical_action_id for any already-scheduled effect, so external systems
    see the same key across run_id rotation.

  - `pending_commits` lives as a TABLE inside the main checkpoint SQLite DB
    (single DB, atomic transactions with state). This module provides the
    connection, schema, and reconciliation primitives, but does NOT own
    the DB handle — the caller (durable_runtime) owns it and passes it in.

  - Three reconciliation modes declared per tool:
      * queryable     — external exposes status-by-key API; reconcile by query.
      * degraded      — write-with-key, no read-back; on resume, do NOT
                        reissue; route to handle_tool_error as 'ambiguous'.
      * at_most_once  — no idempotency guarantees; NEVER auto-reissue on
                        resume; mark failed, operator decides.

  - Tools declare class via @durable_read / @durable_write / @durable_stateful.
    Unclassified tools are a build-time error — the graph-build loop
    validates classification before wiring any tool node.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

logger = logging.getLogger("agent.idempotency")

ReconciliationMode = Literal["queryable", "degraded", "at_most_once"]
CommitStatus = Literal["pending", "committed", "failed", "abandoned"]
DurabilityClass = Literal["read", "write", "stateful"]


# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────

PENDING_COMMITS_SCHEMA = """
CREATE TABLE IF NOT EXISTS pending_commits (
    logical_action_id       TEXT PRIMARY KEY,
    thread_id               TEXT NOT NULL,
    run_id                  TEXT NOT NULL,
    node_name               TEXT NOT NULL,
    call_id                 TEXT NOT NULL,
    tool_name               TEXT NOT NULL,
    payload_hash            TEXT NOT NULL,
    status                  TEXT NOT NULL
                             CHECK(status IN ('pending','committed','failed','abandoned')),
    external_response_hash  TEXT,
    result_json             TEXT,
    reconciliation_mode     TEXT NOT NULL
                             CHECK(reconciliation_mode IN ('queryable','degraded','at_most_once')),
    error_message           TEXT,
    created_at              INTEGER NOT NULL,
    updated_at              INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pending_commits_thread
    ON pending_commits(thread_id, status);
CREATE INDEX IF NOT EXISTS idx_pending_commits_status
    ON pending_commits(status, updated_at);
"""


def initialize_schema(conn: sqlite3.Connection) -> None:
    """Create pending_commits table if absent. Idempotent."""
    conn.executescript(PENDING_COMMITS_SCHEMA)
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# logical_action_id derivation
# ─────────────────────────────────────────────────────────────────────────────

def compute_logical_action_id(
    *,
    thread_id: str,
    node_name: str,
    call_id: str,
    tool_name: str,
    args_hash: str,
    delivery_epoch: int,
) -> str:
    """
    Derive a stable logical_action_id.

    SPEC: this MUST NOT include run_id. The same logical action, issued under
    a replacement run_id after abandonment, must yield the same key so that
    external idempotency protections engage.

    delivery_epoch participates so that a genuinely new effect (e.g., second
    message send in the same node after state advances) gets a fresh key.
    """
    material = "|".join([
        thread_id,
        node_name,
        call_id,
        tool_name,
        args_hash,
        str(delivery_epoch),
    ])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def hash_args(args: Any) -> str:
    """Stable hash of tool arguments — canonical JSON, sha256."""
    try:
        canonical = json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        canonical = repr(args)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def hash_payload(payload: Any) -> str:
    """Hash arbitrary payload for audit (stored on commit)."""
    return hash_args(payload)  # same impl, distinct intent


# ─────────────────────────────────────────────────────────────────────────────
# Commit log operations
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PendingCommit:
    logical_action_id: str
    thread_id: str
    run_id: str
    node_name: str
    call_id: str
    tool_name: str
    payload_hash: str
    status: CommitStatus
    reconciliation_mode: ReconciliationMode
    external_response_hash: Optional[str] = None
    result_json: Optional[str] = None
    error_message: Optional[str] = None
    created_at: int = field(default_factory=lambda: int(time.time()))
    updated_at: int = field(default_factory=lambda: int(time.time()))
    # Stage B extensions (nullable — Stage A rows have None and bypass fencing).
    resume_chain_id: Optional[str] = None
    resume_generation: Optional[int] = None


class IdempotencyStore:
    """Thin wrapper over the pending_commits table. Caller owns the connection."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        initialize_schema(conn)
        self._stage_b_cols_checked = False
        self._stage_b_cols_present = False

    def _pending_commits_has_stage_b_cols(self) -> bool:
        """Cache the result of the one-time column-existence probe."""
        if not self._stage_b_cols_checked:
            cols = {row[1] for row in self.conn.execute(
                "PRAGMA table_info(pending_commits)"
            ).fetchall()}
            self._stage_b_cols_present = (
                "resume_chain_id" in cols and "resume_generation" in cols
            )
            self._stage_b_cols_checked = True
        return self._stage_b_cols_present

    def get(self, logical_action_id: str) -> Optional[PendingCommit]:
        row = self.conn.execute(
            """SELECT logical_action_id, thread_id, run_id, node_name, call_id,
                      tool_name, payload_hash, status, reconciliation_mode,
                      external_response_hash, result_json, error_message,
                      created_at, updated_at
               FROM pending_commits WHERE logical_action_id = ?""",
            (logical_action_id,),
        ).fetchone()
        if row is None:
            return None
        return PendingCommit(
            logical_action_id=row[0], thread_id=row[1], run_id=row[2],
            node_name=row[3], call_id=row[4], tool_name=row[5],
            payload_hash=row[6], status=row[7], reconciliation_mode=row[8],
            external_response_hash=row[9], result_json=row[10],
            error_message=row[11], created_at=row[12], updated_at=row[13],
        )

    def begin_pending(self, commit: PendingCommit) -> PendingCommit:
        """
        Insert a pending row before issuing the side effect.

        If a row already exists for this logical_action_id, returns the
        existing row without modification. That is the core durability
        primitive: a replay path finds the prior pending/committed row
        instead of creating a new one.
        """
        existing = self.get(commit.logical_action_id)
        if existing is not None:
            logger.debug(
                "idempotency: existing commit for %s (status=%s); replay",
                commit.logical_action_id, existing.status,
            )
            return existing

        # Stage A rows leave resume_chain_id/resume_generation NULL.
        # Stage B rows populate both so the fencing trigger on
        # pending_commits can reject stale-generation writes.
        # The columns exist unconditionally on Stage B schema; on pre-B
        # databases we skip them transparently.
        has_stage_b_cols = self._pending_commits_has_stage_b_cols()
        if has_stage_b_cols:
            self.conn.execute(
                """INSERT INTO pending_commits (
                    logical_action_id, thread_id, run_id, node_name, call_id,
                    tool_name, payload_hash, status, reconciliation_mode,
                    created_at, updated_at,
                    resume_chain_id, resume_generation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?)""",
                (
                    commit.logical_action_id, commit.thread_id, commit.run_id,
                    commit.node_name, commit.call_id, commit.tool_name,
                    commit.payload_hash, commit.reconciliation_mode,
                    commit.created_at, commit.updated_at,
                    commit.resume_chain_id, commit.resume_generation,
                ),
            )
        else:
            self.conn.execute(
                """INSERT INTO pending_commits (
                    logical_action_id, thread_id, run_id, node_name, call_id,
                    tool_name, payload_hash, status, reconciliation_mode,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)""",
                (
                    commit.logical_action_id, commit.thread_id, commit.run_id,
                    commit.node_name, commit.call_id, commit.tool_name,
                    commit.payload_hash, commit.reconciliation_mode,
                    commit.created_at, commit.updated_at,
                ),
            )
        # Intentionally no commit() here — caller is expected to commit
        # the enclosing transaction that also writes the graph checkpoint.
        return commit

    def mark_committed(
        self,
        logical_action_id: str,
        *,
        external_response_hash: Optional[str],
        result_json: Optional[str],
    ) -> None:
        now = int(time.time())
        self.conn.execute(
            """UPDATE pending_commits
               SET status='committed', external_response_hash=?,
                   result_json=?, updated_at=?
               WHERE logical_action_id=?""",
            (external_response_hash, result_json, now, logical_action_id),
        )

    def mark_failed(self, logical_action_id: str, *, error_message: str) -> None:
        now = int(time.time())
        self.conn.execute(
            """UPDATE pending_commits
               SET status='failed', error_message=?, updated_at=?
               WHERE logical_action_id=?""",
            (error_message[:2000], now, logical_action_id),
        )

    def mark_abandoned(self, logical_action_id: str, *, reason: str) -> None:
        now = int(time.time())
        self.conn.execute(
            """UPDATE pending_commits
               SET status='abandoned', error_message=?, updated_at=?
               WHERE logical_action_id=?""",
            (f"ABANDONED: {reason[:1900]}", now, logical_action_id),
        )

    def pending_for_thread(self, thread_id: str) -> list[PendingCommit]:
        rows = self.conn.execute(
            """SELECT logical_action_id, thread_id, run_id, node_name, call_id,
                      tool_name, payload_hash, status, reconciliation_mode,
                      external_response_hash, result_json, error_message,
                      created_at, updated_at
               FROM pending_commits
               WHERE thread_id = ? AND status = 'pending'
               ORDER BY created_at ASC""",
            (thread_id,),
        ).fetchall()
        return [
            PendingCommit(
                logical_action_id=r[0], thread_id=r[1], run_id=r[2],
                node_name=r[3], call_id=r[4], tool_name=r[5],
                payload_hash=r[6], status=r[7], reconciliation_mode=r[8],
                external_response_hash=r[9], result_json=r[10],
                error_message=r[11], created_at=r[12], updated_at=r[13],
            )
            for r in rows
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Durability decorators
# ─────────────────────────────────────────────────────────────────────────────

class ToolRegistrationError(Exception):
    """Raised at graph build time when a tool lacks classification."""


# Global registry populated by decorators, keyed by FUNCTION NAME. Useful
# for decorator introspection and test fakes where tool_name == func_name.
_TOOL_REGISTRY: dict[str, dict[str, Any]] = {}

# Side-channel registry keyed by TOOL NAME (the name used in Hermes
# registry.register(name=...)). This is the AUTHORITATIVE source for
# production. It exists because most Hermes tools register with lambda
# handlers that don't carry the function-level __durability__ attribute;
# the decorator can't reach them. Explicit register_durability() calls
# at tool registration time solve this cleanly.
_TOOL_DURABILITY_BY_NAME: dict[str, dict[str, Any]] = {}


def mcp_key(server_id: str, tool_name: str) -> str:
    """Canonical key for MCP tool classifications: 'mcp:{server}:{tool}'."""
    return f"mcp:{server_id}:{tool_name}"


def register_mcp_durability(
    server_id: str,
    tool_name: str,
    *,
    class_: DurabilityClass,
    reconciliation: Optional[ReconciliationMode] = None,
    idempotency_key_fn: Optional[Callable[..., str]] = None,
    stale_after: Optional[float] = None,
) -> None:
    """
    Declare durability for an MCP tool, keyed by (server_id, tool_name).

    Delegates to register_durability with the composite key so both local
    and MCP classifications live in the same side-channel registry.
    """
    register_durability(
        mcp_key(server_id, tool_name),
        class_=class_,
        reconciliation=reconciliation,
        idempotency_key_fn=idempotency_key_fn,
        stale_after=stale_after,
    )


# MCP runtime default: if a tool_name is queried that starts with 'mcp:' but
# isn't explicitly registered, the graph treats it as write/at_most_once —
# the safest conservative default for cross-process side effects. Operators
# can override by calling register_mcp_durability for that specific tool.
MCP_RUNTIME_DEFAULT: dict[str, Any] = {
    "class": "write",
    "reconciliation": "at_most_once",
    "idempotency_key_fn": None,
    "stale_after": None,
    "_is_runtime_default": True,
}


def get_mcp_classification(server_id: str, tool_name: str) -> dict[str, Any]:
    """
    Return the durability classification for an MCP tool. If no explicit
    registration exists, returns the conservative MCP_RUNTIME_DEFAULT (marked
    with _is_runtime_default so callers can warn about unreviewed usage).
    """
    rec = _TOOL_DURABILITY_BY_NAME.get(mcp_key(server_id, tool_name))
    if rec is not None:
        return rec
    return dict(MCP_RUNTIME_DEFAULT)


def register_durability(
    tool_name: str,
    *,
    class_: DurabilityClass,
    reconciliation: Optional[ReconciliationMode] = None,
    idempotency_key_fn: Optional[Callable[..., str]] = None,
    stale_after: Optional[float] = None,
) -> None:
    """
    Declare a tool's durability classification by TOOL NAME.

    This is the canonical production API — called at module import time
    alongside registry.register(). Independent of handler identity, so it
    works with lambda handlers, wrapped handlers, and everything else.

    Validation: class_ must be 'read' | 'write' | 'stateful'. Writes must
    specify reconciliation ∈ {queryable, degraded, at_most_once}. Stateful
    tools ignore reconciliation.
    """
    if class_ not in ("read", "write", "stateful"):
        raise ToolRegistrationError(
            f"register_durability({tool_name!r}): invalid class {class_!r}"
        )
    if class_ == "write":
        if reconciliation not in ("queryable", "degraded", "at_most_once"):
            raise ToolRegistrationError(
                f"register_durability({tool_name!r}): @durable_write requires "
                f"reconciliation ∈ {{queryable, degraded, at_most_once}}, got {reconciliation!r}"
            )
    _TOOL_DURABILITY_BY_NAME[tool_name] = {
        "class": class_,
        "reconciliation": reconciliation if class_ == "write" else None,
        "idempotency_key_fn": idempotency_key_fn,
        "stale_after": stale_after,
    }


def get_classification_by_tool_name(tool_name: str) -> Optional[dict[str, Any]]:
    """Return the durability record for a tool name, or None if unregistered."""
    return _TOOL_DURABILITY_BY_NAME.get(tool_name)


def registered_tool_names() -> set[str]:
    """Return the set of tool names with a registered durability class."""
    return set(_TOOL_DURABILITY_BY_NAME.keys())


def _durability_registry_clear_for_tests() -> None:
    """Test helper: wipe the side-channel registry."""
    _TOOL_DURABILITY_BY_NAME.clear()


def durable_read(func: Callable) -> Callable:
    """
    Mark a tool as read-only / no external side effect.

    Replay is safe: on resume, cached result from prior turn is used verbatim.
    """
    _TOOL_REGISTRY[func.__name__] = {
        "class": "read",
        "fn": func,
        "reconciliation": None,
        "idempotency_key_fn": None,
        "stale_after": None,
    }
    func.__durability__ = {"class": "read"}  # type: ignore[attr-defined]

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper.__durability__ = {"class": "read"}  # type: ignore[attr-defined]
    return wrapper


def durable_write(
    *,
    reconciliation: ReconciliationMode,
    idempotency_key_fn: Optional[Callable[..., str]] = None,
    stale_after: Optional[float] = None,
):
    """
    Mark a tool as having external side effects.

    Args:
        reconciliation: 'queryable' | 'degraded' | 'at_most_once'.
        idempotency_key_fn: optional custom key generator. If omitted, a
            default key is derived from tool name + argument hash. The key
            is only the args-contribution; the full logical_action_id is
            assembled by the runtime which adds thread_id, node_name, etc.
        stale_after: seconds after which a cached result is considered
            stale and the tool is re-executed on resume. Default None =
            infinite (never stale — consistency > freshness, per spec §10.1).
    """
    if reconciliation not in ("queryable", "degraded", "at_most_once"):
        raise ToolRegistrationError(
            f"invalid reconciliation mode: {reconciliation!r}"
        )

    def decorator(func: Callable) -> Callable:
        _TOOL_REGISTRY[func.__name__] = {
            "class": "write",
            "fn": func,
            "reconciliation": reconciliation,
            "idempotency_key_fn": idempotency_key_fn,
            "stale_after": stale_after,
        }
        func.__durability__ = {  # type: ignore[attr-defined]
            "class": "write",
            "reconciliation": reconciliation,
            "stale_after": stale_after,
        }

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.__durability__ = func.__durability__  # type: ignore[attr-defined]
        return wrapper

    return decorator


def durable_stateful(func: Callable) -> Callable:
    """
    Mark a tool as managing its own durable state via a nested StateGraph.

    The tool MUST expose a `get_child_graph()` method returning a LangGraph
    StateGraph. The parent runtime delegates execution to the child, passing
    a checkpointer scoped to (parent thread_id, logical_action_id).
    """
    if not hasattr(func, "get_child_graph"):
        raise ToolRegistrationError(
            f"@durable_stateful tool {func.__name__!r} must expose get_child_graph()"
        )
    _TOOL_REGISTRY[func.__name__] = {
        "class": "stateful",
        "fn": func,
        "reconciliation": None,
        "idempotency_key_fn": None,
        "stale_after": None,
    }
    func.__durability__ = {"class": "stateful"}  # type: ignore[attr-defined]

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper.__durability__ = func.__durability__  # type: ignore[attr-defined]
    return wrapper


def require_classification(tool_names: list[str]) -> None:
    """
    Call at graph build time. Raises ToolRegistrationError if any tool in
    the list lacks a durability classification.

    Lookup order:
      1. Side-channel registry keyed by tool_name (authoritative — production
         tools declare via register_durability()).
      2. Decorator registry keyed by function name (tests and tools where
         tool_name == function_name).
    """
    missing: list[str] = []
    for n in tool_names:
        if n in _TOOL_DURABILITY_BY_NAME:
            continue
        if n in _TOOL_REGISTRY:
            continue
        missing.append(n)
    if missing:
        raise ToolRegistrationError(
            "graph build: tools missing durability classification: "
            + ", ".join(missing)
            + ". Every tool attached to [execute_tools] must have either a "
            "register_durability(tool_name=...) call or a @durable_* decorator "
            "on the handler whose __name__ matches the tool name."
        )


def get_classification(tool_name: str) -> Optional[dict[str, Any]]:
    entry = _TOOL_REGISTRY.get(tool_name)
    if entry is None:
        return None
    # Return a shallow copy without the function reference
    return {k: v for k, v in entry.items() if k != "fn"}


def _registry_clear_for_tests() -> None:
    """Test helper: clear the global registry between runs."""
    _TOOL_REGISTRY.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Reconciliation (called from [reconcile_pending_commits] node on resume)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReconciliationResult:
    logical_action_id: str
    action: Literal["committed", "reissue", "ambiguous", "failed"]
    note: str


class QueryableReconciler:
    """
    Interface a @durable_write(reconciliation='queryable') tool must expose
    so the runtime can ask 'did my side effect actually commit?' on resume.

    Implement either as a class with these methods, or pass callables to
    `reconcile_queryable()` directly.
    """

    def probe(self, *, logical_action_id: str, tool_name: str,
              payload_hash: str) -> Literal["committed", "not_found", "ambiguous"]:
        raise NotImplementedError


def reconcile_pending_commits(
    store: IdempotencyStore,
    thread_id: str,
    *,
    reconcilers: dict[str, QueryableReconciler],
) -> list[ReconciliationResult]:
    """
    Walk all pending commits for a thread and resolve them per mode:

      queryable     → call the reconciler; apply its verdict.
      degraded      → do NOT query, do NOT reissue; mark 'ambiguous' for operator.
      at_most_once  → mark 'failed'; route to handle_tool_error; do NOT reissue.

    Does NOT commit the DB — caller wraps this in a transaction with the
    graph state update.
    """
    results: list[ReconciliationResult] = []
    for commit in store.pending_for_thread(thread_id):
        if commit.reconciliation_mode == "queryable":
            reconciler = reconcilers.get(commit.tool_name)
            if reconciler is None:
                results.append(ReconciliationResult(
                    commit.logical_action_id, "ambiguous",
                    f"no reconciler registered for queryable tool {commit.tool_name!r}",
                ))
                continue
            verdict = reconciler.probe(
                logical_action_id=commit.logical_action_id,
                tool_name=commit.tool_name,
                payload_hash=commit.payload_hash,
            )
            if verdict == "committed":
                store.mark_committed(
                    commit.logical_action_id,
                    external_response_hash=None, result_json=None,
                )
                results.append(ReconciliationResult(
                    commit.logical_action_id, "committed",
                    "queryable reconciler confirmed external commit",
                ))
            elif verdict == "not_found":
                results.append(ReconciliationResult(
                    commit.logical_action_id, "reissue",
                    "queryable reconciler reports no external record — reissue",
                ))
            else:  # ambiguous
                results.append(ReconciliationResult(
                    commit.logical_action_id, "ambiguous",
                    "queryable reconciler returned ambiguous status",
                ))

        elif commit.reconciliation_mode == "degraded":
            results.append(ReconciliationResult(
                commit.logical_action_id, "ambiguous",
                "degraded mode: cannot query external; operator must resolve",
            ))

        elif commit.reconciliation_mode == "at_most_once":
            store.mark_failed(
                commit.logical_action_id,
                error_message="at_most_once tool: crash during effect; not reissued",
            )
            results.append(ReconciliationResult(
                commit.logical_action_id, "failed",
                "at_most_once: marked failed, not reissued",
            ))

        else:
            # Defensive: unreachable given schema CHECK constraint.
            raise RuntimeError(
                f"unknown reconciliation mode: {commit.reconciliation_mode!r}"
            )

    return results
