"""
durable_runtime.py — the LangGraph-based durable agent loop for Hermes.

Implements spec v0.3: a StateGraph wrapping the Hermes agent loop, with:

  - AgentState TypedDict (§4.3)
  - Nodes: load_and_compress_context, reconcile_pending_commits, call_model,
    execute_tools, respond, handle_llm_error, handle_tool_error,
    handle_delivery_error, and terminal routing.
  - Sync + WAL-style response capture for call_model (§4.10).
  - Cursor-based exactly-once respond (§4.11).
  - Checkpointer with integrity_check + degraded-read-only on failure (§4.7).
  - Concurrent + rate-limited + prioritized resume (§4.9).
  - Feature-flagged: integration with the live gateway is gated by
    `durable_runtime: true/false` in config.yaml. This module is safe to
    import without enabling the flag — no side effects on import.

Phase 1 scope per §5: wire the graph with ONE side-effecting tool and
prove the 7 crash scenarios. Full tool audit is Phase 2.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, Optional, TypedDict

logger = logging.getLogger("agent.durable_runtime")

# LangGraph imports deferred behind a try/except so this module can be imported
# in environments where langgraph isn't installed (e.g., Windows clients until
# Phase 4 flips them on). Runtime use without the library will raise clearly.
try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.errors import GraphInterrupt
    _LANGGRAPH_AVAILABLE = True
except ImportError as _import_err:
    StateGraph = None  # type: ignore[misc,assignment]
    START = None        # type: ignore[misc,assignment]
    END = None          # type: ignore[misc,assignment]
    SqliteSaver = None  # type: ignore[misc,assignment]
    GraphInterrupt = Exception  # type: ignore[misc,assignment]
    _LANGGRAPH_AVAILABLE = False
    _LANGGRAPH_IMPORT_ERROR = _import_err

# Sibling modules — same package
try:
    from . import idempotency as idem
    from . import state_migrations as migs
    from . import error_classification as errclass
except ImportError:
    # Direct execution / test harness
    import idempotency as idem  # type: ignore[no-redef]
    import state_migrations as migs  # type: ignore[no-redef]
    import error_classification as errclass  # type: ignore[no-redef]


# ─────────────────────────────────────────────────────────────────────────────
# State shape (spec §4.3)
# ─────────────────────────────────────────────────────────────────────────────

class ChatMessage(TypedDict, total=False):
    role: str            # "user" | "assistant" | "system" | "tool"
    content: str
    tool_call_id: Optional[str]
    name: Optional[str]


class ToolCall(TypedDict, total=False):
    call_id: str
    tool_name: str
    args: dict
    logical_action_id: Optional[str]  # assigned before first execution


class ToolResult(TypedDict, total=False):
    call_id: str
    tool_name: str
    ok: bool
    result: Any
    error: Optional[str]


class AgentState(TypedDict, total=False):
    # Identity
    thread_id: str
    run_id: str
    graph_version: int
    delivery_epoch: int

    # Conversation
    session_key: str
    messages: list[ChatMessage]
    compressed_context_blob_id: Optional[str]
    agent_profile: str
    model: str

    # Tool execution
    pending_tool_calls: list[ToolCall]
    in_flight_tool_calls: list[ToolCall]
    completed_tool_calls: list[ToolResult]

    # call_model WAL capture
    last_model_request_hash: Optional[str]
    last_model_response_raw: Optional[dict]
    last_model_version: Optional[str]

    # Response
    response_cursor: int
    response_message_id: Optional[str]

    # Error tracking
    llm_retry_count: int
    tool_retry_count: int
    last_error_class: Optional[str]

    # Metadata
    turn_count: int
    started_at: str
    last_checkpoint_at: str
    terminal_state: Optional[str]  # "SUCCESS" | "FAIL" | "ABANDONED" | None


# ─────────────────────────────────────────────────────────────────────────────
# Defaults and paths
# ─────────────────────────────────────────────────────────────────────────────

HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
CHECKPOINTS_DB = HERMES_HOME / "data" / "langgraph-checkpoints.db"
CONTEXT_BLOBS_DIR = HERMES_HOME / "data" / "context-blobs"
CONTEXT_BLOBS_GC_DIR = HERMES_HOME / "data" / "context-blobs-gc"
ARCHIVE_DIR = HERMES_HOME / "data" / "langgraph-archive"
DEGRADED_MARKER = HERMES_HOME / "state" / "durable-runtime-degraded.json"

DEFAULT_LLM_RETRY_BUDGET = 3
DEFAULT_TOOL_RETRY_BUDGET = 3
DEFAULT_DELIVERY_RETRY_BUDGET = 3
RECOVERY_READY_DEADLINE_S = 30
RECOVERY_CONCURRENCY = 5
RECOVERY_OUTBOUND_RPS = 10.0


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointer setup with integrity check (§4.7)
# ─────────────────────────────────────────────────────────────────────────────

class DegradedReadOnlyError(RuntimeError):
    """Gateway must not start new runs — DB integrity failure."""


@dataclass
class CheckpointDb:
    """Wraps a SQLite connection with integrity-check and schema init."""
    path: Path
    conn: sqlite3.Connection
    saver: Any  # SqliteSaver, typed loosely to allow lazy-import fallback

    @classmethod
    def open(cls, db_path: Path) -> "CheckpointDb":
        """
        Open (or create) the checkpoint DB. Runs PRAGMA integrity_check.
        On failure, raises DegradedReadOnlyError — caller must refuse to
        start new runs and surface operator alert.
        """
        if not _LANGGRAPH_AVAILABLE:
            raise RuntimeError(
                "langgraph not importable — cannot open checkpoint DB. "
                f"Install langgraph first: pip install langgraph langgraph-checkpoint-sqlite. "
                f"Original error: {_LANGGRAPH_IMPORT_ERROR!r}"
            )
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
        try:
            # Integrity check BEFORE attaching SqliteSaver, and BEFORE any
            # write-side pragma, so corrupted/non-sqlite files surface as
            # a DegradedReadOnly condition rather than a generic init error.
            try:
                row = conn.execute("PRAGMA integrity_check").fetchone()
                verdict = row[0] if row else "unknown"
            except sqlite3.DatabaseError as db_err:
                # Non-sqlite file or severely corrupted — treat as integrity failure.
                verdict = f"sqlite_database_error: {db_err}"
            if verdict != "ok":
                # Set WAL only for healthy DBs.
                pass
            else:
                conn.execute("PRAGMA journal_mode=WAL")
            if verdict != "ok":
                corrupted_copy = db_path.with_name(
                    f"{db_path.name}.corrupted-{int(time.time())}"
                )
                logger.error(
                    "durable_runtime: DB integrity check FAILED (%s). Moving to %s. "
                    "Entering degraded read-only — no new runs will start.",
                    verdict, corrupted_copy,
                )
                conn.close()
                try:
                    shutil.copy2(db_path, corrupted_copy)
                except OSError as copy_err:
                    logger.error("durable_runtime: failed to copy corrupted DB aside: %s", copy_err)
                _write_degraded_marker(verdict, str(corrupted_copy))
                raise DegradedReadOnlyError(
                    f"checkpoint DB integrity check failed: {verdict}. "
                    f"Corrupted file preserved at {corrupted_copy}. "
                    "Gateway refuses to start new runs until operator resolves."
                )
            # Good — init schema for pending_commits and migration_log.
            idem.initialize_schema(conn)
            migs.initialize_schema(conn)
        except DegradedReadOnlyError:
            raise
        except Exception as exc:
            conn.close()
            raise RuntimeError(f"failed to initialize checkpoint DB: {exc}") from exc

        saver = SqliteSaver(conn)
        return cls(path=db_path, conn=conn, saver=saver)

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass


def _write_degraded_marker(verdict: str, corrupted_path: str) -> None:
    DEGRADED_MARKER.parent.mkdir(parents=True, exist_ok=True)
    DEGRADED_MARKER.write_text(json.dumps({
        "entered_at": int(time.time()),
        "verdict": verdict,
        "corrupted_db": corrupted_path,
        "resolution_required": True,
    }, indent=2), encoding="utf-8")


def clear_degraded_marker() -> None:
    """Operator-driven: called after manual resolution."""
    try:
        DEGRADED_MARKER.unlink()
    except FileNotFoundError:
        pass


def is_degraded() -> bool:
    return DEGRADED_MARKER.exists()


# ─────────────────────────────────────────────────────────────────────────────
# Model client interface
# ─────────────────────────────────────────────────────────────────────────────

class ModelClient:
    """
    Abstract interface over the underlying LLM provider call.

    The runtime is agnostic to SDK details. Implementations are provided by
    adapters (OpenAI-compatible, Anthropic, etc). Phase 1 uses a simple
    synchronous callable.
    """

    def call(self, *, model: str, messages: list[ChatMessage],
             tools: Optional[list[dict]] = None) -> dict:
        """
        Return the raw provider response. The runtime stores this verbatim
        into state.last_model_response_raw before any downstream node runs,
        so replay does NOT re-call the provider.
        """
        raise NotImplementedError


class DeliveryAdapter:
    """Platform-facing delivery interface used by [respond]."""

    def send_or_edit(self, *, message_id: str, content: str,
                     cursor: int) -> dict:
        """
        Send (first cursor) or edit (resumed cursor) a message.
        Returns a dict with {'ok': bool, 'message_id': str, 'bytes_sent': int}.
        """
        raise NotImplementedError

    def is_message_sent(self, message_id: str) -> bool:
        """For non-streaming: probe whether a prior message_id was delivered."""
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# Runtime: wires state + services + graph together
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DurableRuntime:
    checkpoint_db: CheckpointDb
    model_client: ModelClient
    delivery_adapter: DeliveryAdapter
    tool_dispatcher: Callable[[ToolCall], ToolResult]
    tool_names: list[str]  # for classification check at build time
    # Reconciler registry: tool_name -> QueryableReconciler
    reconcilers: dict[str, idem.QueryableReconciler] = field(default_factory=dict)
    # Retry budgets (overridable per profile)
    llm_retry_budget: int = DEFAULT_LLM_RETRY_BUDGET
    tool_retry_budget: int = DEFAULT_TOOL_RETRY_BUDGET
    delivery_retry_budget: int = DEFAULT_DELIVERY_RETRY_BUDGET

    def __post_init__(self):
        # Enforce tool classification at build time (§4.5).
        idem.require_classification(self.tool_names)
        # Validate migrations before any thread can resume (§4.8).
        migs.validate_all_migrations()

    @property
    def idem_store(self) -> idem.IdempotencyStore:
        return idem.IdempotencyStore(self.checkpoint_db.conn)

    # ─────────────────────────────────────────────────────────────────────
    # Graph nodes
    # ─────────────────────────────────────────────────────────────────────

    def node_load_and_compress_context(self, state: AgentState) -> AgentState:
        """
        Load history + memory + SOUL; run compression INSIDE the graph so the
        compressed context is deterministic on replay. Store content-addressed
        blob id. Actual Hermes memory loading is delegated to existing code
        in Phase 3; Phase 1 just stamps the blob id.
        """
        messages = state.get("messages", [])
        # Build a canonical representation of what would be compressed.
        payload = {
            "messages": messages,
            "agent_profile": state.get("agent_profile"),
            "model": state.get("model"),
        }
        blob_id = _write_context_blob(payload)
        new_state: AgentState = {**state, "compressed_context_blob_id": blob_id}
        new_state["last_checkpoint_at"] = _now_iso()
        return new_state

    def node_reconcile_pending_commits(self, state: AgentState) -> AgentState:
        """
        On resume, walk pending commits for this thread and resolve per mode.
        On fresh runs this is a no-op (no pending commits).
        """
        thread_id = state["thread_id"]
        store = self.idem_store
        results = idem.reconcile_pending_commits(
            store, thread_id, reconcilers=self.reconcilers,
        )
        self.checkpoint_db.conn.commit()
        if results:
            logger.info(
                "durable_runtime: reconciled %d pending commits for thread %s: %s",
                len(results), thread_id,
                [(r.logical_action_id[:12], r.action) for r in results],
            )
            # If any commit is ambiguous, mark last_error_class so the graph
            # routes to handle_tool_error for operator attention.
            if any(r.action in ("ambiguous", "failed") for r in results):
                return {**state, "last_error_class": "permanent"}
        return state

    def node_call_model(self, state: AgentState) -> AgentState:
        """
        Synchronous durability with write-ahead capture: the raw response is
        persisted to state BEFORE the graph routes to any downstream node.

        On replay: if last_model_request_hash matches current request and the
        captured response is present, skip the provider call.
        """
        messages = list(state.get("messages", []))
        request_hash = _hash_request(
            state.get("model", ""),
            messages,
            state.get("agent_profile", ""),
        )

        # Replay fast path
        if (state.get("last_model_request_hash") == request_hash
                and state.get("last_model_response_raw") is not None):
            logger.debug("durable_runtime: call_model replay from captured response")
            response = state["last_model_response_raw"]
        else:
            try:
                response = self.model_client.call(
                    model=state.get("model", ""),
                    messages=messages,
                )
            except Exception as exc:
                error_class = errclass.classify_exception(exc)
                logger.warning(
                    "durable_runtime: call_model failed: %s (%s)",
                    exc, error_class,
                )
                return {
                    **state,
                    "last_error_class": error_class,
                    "llm_retry_count": state.get("llm_retry_count", 0) + 1,
                }

        # Capture BEFORE any downstream routing — spec §4.10.
        new_state: AgentState = {
            **state,
            "last_model_request_hash": request_hash,
            "last_model_response_raw": response,
            "last_model_version": response.get("model") if isinstance(response, dict) else None,
            "last_checkpoint_at": _now_iso(),
        }

        # Parse the response into the state's message list and tool call queue.
        parsed = _parse_model_response(response)
        assistant_msg = parsed.get("assistant_message")
        tool_calls = parsed.get("tool_calls", [])

        if assistant_msg:
            new_state["messages"] = messages + [assistant_msg]

        if tool_calls:
            # Assign logical_action_id to every scheduled tool call NOW.
            # This is the critical durability primitive from §4.5.1 — the id
            # is persistent across run_id rotation.
            delivery_epoch = state.get("delivery_epoch", 0)
            for tc in tool_calls:
                tc["logical_action_id"] = idem.compute_logical_action_id(
                    thread_id=state["thread_id"],
                    node_name="execute_tools",
                    call_id=tc["call_id"],
                    tool_name=tc["tool_name"],
                    args_hash=idem.hash_args(tc.get("args", {})),
                    delivery_epoch=delivery_epoch,
                )
            new_state["pending_tool_calls"] = tool_calls

        new_state["turn_count"] = state.get("turn_count", 0) + 1
        return new_state

    def node_execute_tools(self, state: AgentState) -> AgentState:
        """
        Dispatch pending tool calls through the idempotency contract.

        For each call:
          1. Check if logical_action_id already has a committed record
             (replay safety). If yes, use cached result.
          2. Otherwise, write pending_commit row, execute tool, update
             committed on success. State + commit share the enclosing DB
             transaction.
        """
        store = self.idem_store
        pending = list(state.get("pending_tool_calls", []))
        completed = list(state.get("completed_tool_calls", []))
        in_flight = list(state.get("in_flight_tool_calls", []))
        run_id = state["run_id"]
        thread_id = state["thread_id"]

        messages = list(state.get("messages", []))
        tool_error_class: Optional[str] = None

        for tc in pending:
            lai = tc["logical_action_id"]
            classification = idem.get_classification(tc["tool_name"]) or {}
            tool_class = classification.get("class")

            existing = store.get(lai)
            # Fast path: prior committed record → reuse cached result.
            if existing is not None and existing.status == "committed":
                logger.info(
                    "durable_runtime: tool %s (lai=%s) replaying committed result",
                    tc["tool_name"], lai[:12],
                )
                cached = (json.loads(existing.result_json)
                          if existing.result_json else None)
                result: ToolResult = {
                    "call_id": tc["call_id"],
                    "tool_name": tc["tool_name"],
                    "ok": True,
                    "result": cached,
                }
                completed.append(result)
                messages.append(_tool_result_message(result))
                continue

            # Fast path: prior failed-reconciliation → route to handler.
            if existing is not None and existing.status == "failed":
                logger.warning(
                    "durable_runtime: tool %s (lai=%s) prior commit failed; "
                    "routing to handle_tool_error",
                    tc["tool_name"], lai[:12],
                )
                tool_error_class = "permanent"
                result = {
                    "call_id": tc["call_id"],
                    "tool_name": tc["tool_name"],
                    "ok": False,
                    "error": existing.error_message or "prior commit failed",
                }
                completed.append(result)
                messages.append(_tool_result_message(result))
                continue

            # @durable_read: no commit log needed.
            if tool_class == "read":
                try:
                    res = self.tool_dispatcher(tc)
                    completed.append(res)
                    messages.append(_tool_result_message(res))
                except Exception as exc:
                    ec = errclass.classify_exception(exc, tool_name=tc["tool_name"])
                    tool_error_class = ec if tool_error_class != "permanent" else tool_error_class
                    if ec == "permanent":
                        tool_error_class = "permanent"
                    result = {
                        "call_id": tc["call_id"],
                        "tool_name": tc["tool_name"],
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    completed.append(result)
                    messages.append(_tool_result_message(result))
                continue

            # @durable_write or @durable_stateful: full idempotency path.
            reconciliation_mode = classification.get("reconciliation") or "at_most_once"
            commit = idem.PendingCommit(
                logical_action_id=lai,
                thread_id=thread_id,
                run_id=run_id,
                node_name="execute_tools",
                call_id=tc["call_id"],
                tool_name=tc["tool_name"],
                payload_hash=idem.hash_payload(tc.get("args", {})),
                status="pending",
                reconciliation_mode=reconciliation_mode,
            )
            store.begin_pending(commit)
            # Commit the pending row BEFORE the effect — so a crash between
            # pending-write and effect-call still leaves a row the reconciler
            # can see. This is the atomicity requirement of §4.5.
            self.checkpoint_db.conn.commit()
            in_flight.append(tc)

            try:
                res = self.tool_dispatcher(tc)
                # Mark committed.
                result_json = json.dumps(res.get("result"), default=str) if res.get("ok") else None
                external_response_hash = idem.hash_payload(res) if res.get("ok") else None
                store.mark_committed(
                    lai,
                    external_response_hash=external_response_hash,
                    result_json=result_json,
                )
                self.checkpoint_db.conn.commit()
                in_flight.remove(tc)
                completed.append(res)
                messages.append(_tool_result_message(res))
            except Exception as exc:
                ec = errclass.classify_exception(exc, tool_name=tc["tool_name"])
                store.mark_failed(lai, error_message=f"{type(exc).__name__}: {exc}")
                self.checkpoint_db.conn.commit()
                try:
                    in_flight.remove(tc)
                except ValueError:
                    pass
                if ec == "permanent":
                    tool_error_class = "permanent"
                elif tool_error_class != "permanent":
                    tool_error_class = ec
                result = {
                    "call_id": tc["call_id"],
                    "tool_name": tc["tool_name"],
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                completed.append(result)
                messages.append(_tool_result_message(result))

        new_state: AgentState = {
            **state,
            "messages": messages,
            "pending_tool_calls": [],
            "in_flight_tool_calls": in_flight,
            "completed_tool_calls": completed,
            "last_checkpoint_at": _now_iso(),
        }
        if tool_error_class is not None:
            new_state["last_error_class"] = tool_error_class
            new_state["tool_retry_count"] = state.get("tool_retry_count", 0) + 1
        return new_state

    def node_respond(self, state: AgentState) -> AgentState:
        """
        Cursor-based exactly-once delivery (§4.11). Deterministic message_id
        from (thread_id, logical_action_id_for_respond, turn_count).
        """
        messages = state.get("messages", [])
        # Final assistant content = the last assistant message.
        last_assistant = next(
            (m for m in reversed(messages) if m.get("role") == "assistant"),
            None,
        )
        content = (last_assistant or {}).get("content", "") if last_assistant else ""

        # logical_action_id for this respond node
        lai = idem.compute_logical_action_id(
            thread_id=state["thread_id"],
            node_name="respond",
            call_id="respond",
            tool_name="respond",
            args_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            delivery_epoch=state.get("delivery_epoch", 0),
        )
        message_id = state.get("response_message_id") or f"msg_{lai[:16]}"
        cursor = state.get("response_cursor", 0)

        try:
            res = self.delivery_adapter.send_or_edit(
                message_id=message_id,
                content=content,
                cursor=cursor,
            )
            if not res.get("ok"):
                return {
                    **state,
                    "last_error_class": "transient",
                    "response_message_id": message_id,
                }
            return {
                **state,
                "response_message_id": message_id,
                "response_cursor": int(res.get("bytes_sent", cursor)),
                "terminal_state": "SUCCESS",
                "last_checkpoint_at": _now_iso(),
            }
        except Exception as exc:
            ec = errclass.classify_exception(exc, tool_name="respond")
            return {
                **state,
                "last_error_class": ec,
                "response_message_id": message_id,
            }

    def node_handle_llm_error(self, state: AgentState) -> AgentState:
        """Bounded retry on transient; terminal FAIL on permanent/budget exceeded."""
        ec = state.get("last_error_class", "unknown")
        if ec != "transient":
            return {**state, "terminal_state": "FAIL"}
        if state.get("llm_retry_count", 0) >= self.llm_retry_budget:
            return {**state, "terminal_state": "FAIL"}
        # Clear error so call_model is re-entered.
        return {**state, "last_error_class": None}

    def node_handle_tool_error(self, state: AgentState) -> AgentState:
        """Feed tool errors back into messages; model decides next step."""
        ec = state.get("last_error_class", "unknown")
        if ec == "permanent":
            # Permanent tool errors: terminal FAIL (spec §4.2).
            return {**state, "terminal_state": "FAIL"}
        if state.get("tool_retry_count", 0) >= self.tool_retry_budget:
            return {**state, "terminal_state": "FAIL"}
        return {**state, "last_error_class": None}

    def node_handle_delivery_error(self, state: AgentState) -> AgentState:
        ec = state.get("last_error_class", "unknown")
        if ec == "permanent":
            return {**state, "terminal_state": "FAIL"}
        if state.get("tool_retry_count", 0) >= self.delivery_retry_budget:
            return {**state, "terminal_state": "FAIL"}
        return {
            **state,
            "last_error_class": None,
            "tool_retry_count": state.get("tool_retry_count", 0) + 1,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Routers
    # ─────────────────────────────────────────────────────────────────────

    def route_after_call_model(self, state: AgentState) -> str:
        ec = state.get("last_error_class")
        if ec in ("permanent", "unknown"):
            return "handle_llm_error"
        if ec == "transient":
            return "handle_llm_error"
        if state.get("pending_tool_calls"):
            return "execute_tools"
        return "respond"

    def route_after_execute_tools(self, state: AgentState) -> str:
        ec = state.get("last_error_class")
        if ec == "permanent":
            return "handle_tool_error"
        if ec == "transient":
            return "handle_tool_error"
        # Success path: loop back to the model with tool results in messages.
        return "call_model"

    def route_after_respond(self, state: AgentState) -> str:
        if state.get("terminal_state") == "SUCCESS":
            return END  # type: ignore[return-value]
        return "handle_delivery_error"

    def route_after_error_handler(self, state: AgentState, *, retry_target: str) -> str:
        if state.get("terminal_state") == "FAIL":
            return END  # type: ignore[return-value]
        return retry_target

    # ─────────────────────────────────────────────────────────────────────
    # Graph construction
    # ─────────────────────────────────────────────────────────────────────

    def build_graph(self):
        """Construct the StateGraph and compile with the checkpointer."""
        if not _LANGGRAPH_AVAILABLE:
            raise RuntimeError(
                "langgraph not importable — cannot build graph. "
                f"Install langgraph first. Error: {_LANGGRAPH_IMPORT_ERROR!r}"
            )

        g = StateGraph(AgentState)

        g.add_node("load_and_compress_context", self.node_load_and_compress_context)
        g.add_node("reconcile_pending_commits", self.node_reconcile_pending_commits)
        g.add_node("call_model", self.node_call_model)
        g.add_node("execute_tools", self.node_execute_tools)
        g.add_node("respond", self.node_respond)
        g.add_node("handle_llm_error", self.node_handle_llm_error)
        g.add_node("handle_tool_error", self.node_handle_tool_error)
        g.add_node("handle_delivery_error", self.node_handle_delivery_error)

        g.add_edge(START, "load_and_compress_context")
        g.add_edge("load_and_compress_context", "reconcile_pending_commits")
        g.add_edge("reconcile_pending_commits", "call_model")

        g.add_conditional_edges("call_model", self.route_after_call_model, {
            "handle_llm_error": "handle_llm_error",
            "execute_tools": "execute_tools",
            "respond": "respond",
        })

        g.add_conditional_edges("execute_tools", self.route_after_execute_tools, {
            "handle_tool_error": "handle_tool_error",
            "call_model": "call_model",
        })

        g.add_conditional_edges("respond", self.route_after_respond, {
            END: END,
            "handle_delivery_error": "handle_delivery_error",
        })

        g.add_conditional_edges(
            "handle_llm_error",
            lambda s: self.route_after_error_handler(s, retry_target="call_model"),
            {END: END, "call_model": "call_model"},
        )
        g.add_conditional_edges(
            "handle_tool_error",
            lambda s: self.route_after_error_handler(s, retry_target="call_model"),
            {END: END, "call_model": "call_model"},
        )
        g.add_conditional_edges(
            "handle_delivery_error",
            lambda s: self.route_after_error_handler(s, retry_target="respond"),
            {END: END, "respond": "respond"},
        )

        return g.compile(checkpointer=self.checkpoint_db.saver)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _hash_request(model: str, messages: list[ChatMessage], profile: str) -> str:
    payload = {"model": model, "messages": messages, "profile": profile}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _write_context_blob(payload: Any) -> str:
    """Content-addressed blob storage (§4.6). Immutable post-creation."""
    canonical = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    blob_id = hashlib.sha256(canonical).hexdigest()
    out_dir = CONTEXT_BLOBS_DIR / blob_id[:2]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{blob_id}.json"
    if not out_path.exists():
        # Write then chmod — preserves immutability from any non-root reader.
        out_path.write_bytes(canonical)
        try:
            os.chmod(out_path, 0o400)
        except OSError:
            pass  # best-effort on Windows
    return blob_id


def _parse_model_response(response: dict) -> dict:
    """
    Extract assistant message + tool calls from a provider response.

    Supports OpenAI-compatible format. Adapters for Anthropic-format
    responses normalize into this shape before reaching the runtime.
    """
    if not isinstance(response, dict):
        return {"assistant_message": None, "tool_calls": []}
    choices = response.get("choices") or []
    if not choices:
        return {"assistant_message": None, "tool_calls": []}
    msg = choices[0].get("message") or {}
    content = msg.get("content") or ""
    tool_calls_raw = msg.get("tool_calls") or []
    tool_calls: list[ToolCall] = []
    for tc in tool_calls_raw:
        fn = tc.get("function") or {}
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        tool_calls.append({
            "call_id": tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
            "tool_name": fn.get("name") or "unknown",
            "args": args,
        })
    assistant_message: ChatMessage = {"role": "assistant", "content": content}
    return {"assistant_message": assistant_message, "tool_calls": tool_calls}


def _tool_result_message(result: ToolResult) -> ChatMessage:
    body = result.get("result") if result.get("ok") else result.get("error")
    return {
        "role": "tool",
        "content": json.dumps(body, default=str) if not isinstance(body, str) else body,
        "tool_call_id": result.get("call_id"),
        "name": result.get("tool_name"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Resume on startup (§4.9) — rate-limited, prioritized, background
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ResumeReport:
    fresh_resumed: int = 0
    stale_pending_review: int = 0
    interrupted_resumed: int = 0
    migration_errors: int = 0
    reconciliation_failures: int = 0


def resume_pending_threads_background(
    runtime: DurableRuntime,
    *,
    ready_event: threading.Event,
    deadline_s: float = RECOVERY_READY_DEADLINE_S,
    concurrency: int = RECOVERY_CONCURRENCY,
    rps: float = RECOVERY_OUTBOUND_RPS,
) -> ResumeReport:
    """
    Signal gateway-ready within `deadline_s` regardless of backlog, then
    continue recovery in the background with rate-limit + priority.

    This is a synchronous scaffold for Phase 1 — in Phase 3 it becomes
    asyncio-native. For Phase 1 tests we can call this directly and
    observe the ResumeReport; the ready_event is set immediately so the
    "gateway ready within 30s" contract is trivially satisfied.
    """
    # Phase 1 minimal: stub that marks ready, surfaces the DB state, and
    # returns a report. A full implementation (priority queue, rate-limit,
    # background threads) follows once Phase 1 tests confirm the semantics.
    ready_event.set()
    # Future: walk checkpointer for non-terminal threads, classify by
    # interrupt / fresh / stale, schedule resumes honoring concurrency/rps.
    return ResumeReport()
