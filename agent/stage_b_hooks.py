"""
agent/stage_b_hooks.py — the single seam between run_conversation and the
Stage B durability machinery.

Designed so run_conversation's patches are MINIMAL: one import, three
single-line checkpoint calls per turn, one dispatch_durable call per tool.
Everything else lives here.

All public functions are SAFE to call when Stage B is disabled — they
return quickly without touching the DB, so a flag-off path pays zero
Stage B cost.

Integration pattern in run_conversation:

    # At top of function:
    from agent import stage_b_hooks

    stage_b_state = stage_b_hooks.begin_run(
        self, thread_id, session_id, resume_state=resume_state,
    )
    if stage_b_state.resumed:
        # Restore local variables from checkpoint
        api_call_count = stage_b_state.api_call_count
        messages      = stage_b_state.messages
        # ... etc

    while api_call_count < max_turns:
        # Phase 1 — pre_call
        stage_b_hooks.checkpoint(self, stage_b_state, phase=1)

        response = self._interruptible_api_call(...)  # or streaming

        # Phase 2 — post_response (response captured)
        stage_b_hooks.checkpoint(self, stage_b_state, phase=2,
                                 response_body=response_raw)

        # Tools dispatched via wrapped call
        self._execute_tool_calls(...)  # uses _dispatch_durable internally

        # Phase 3 — post_tools
        stage_b_hooks.checkpoint(self, stage_b_state, phase=3)

        api_call_count += 1

    # At end:
    stage_b_hooks.end_run(self, stage_b_state, terminal_state='SUCCESS')
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

# Lazy-imported siblings (keeps this module importable when Stage B is off).
_ds = None       # durable_state
_di = None       # durable_integration
_dr = None       # durable_resume
_idem = None     # idempotency

logger = logging.getLogger("agent.stage_b_hooks")


# ─────────────────────────────────────────────────────────────────────────────
# Lazy-load helpers — avoid import cost when flag is off
# ─────────────────────────────────────────────────────────────────────────────

def _load_deps():
    global _ds, _di, _dr, _idem
    if _ds is None:
        try:
            from . import durable_state as _ds_mod
            from . import durable_integration as _di_mod
            from . import durable_resume as _dr_mod
            from . import idempotency as _idem_mod
        except ImportError:
            import durable_state as _ds_mod  # type: ignore[no-redef]
            import durable_integration as _di_mod  # type: ignore[no-redef]
            import durable_resume as _dr_mod  # type: ignore[no-redef]
            import idempotency as _idem_mod  # type: ignore[no-redef]
        _ds = _ds_mod
        _di = _di_mod
        _dr = _dr_mod
        _idem = _idem_mod


def _cfg() -> dict:
    """Read gateway config via durable_integration.is_stage_b_enabled's same source."""
    # Pull via the same _load_gateway_config pattern the gateway uses.
    try:
        from gateway.run import _load_gateway_config
        return _load_gateway_config() or {}
    except ImportError:
        return {}


def is_enabled() -> bool:
    """Fast check called at every checkpoint point. Must be cheap."""
    _load_deps()
    return _di.is_stage_b_enabled(_cfg())


# ─────────────────────────────────────────────────────────────────────────────
# Run state tracking
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class StageBRunState:
    """
    State carried through a single run_conversation invocation.

    Attributes mirror StageBState keys but are held as a dataclass for
    cheaper attribute access from inside the hot loop. serialize() produces
    the JSON blob for checkpoint writes.
    """
    enabled: bool = False             # Stage B on for this run?
    resumed: bool = False             # did we resume from checkpoint?
    thread_id: str = ""
    session_id: str = ""
    run_id: str = ""
    resume_chain_id: str = ""
    resume_generation: int = 0
    agent_profile: str = ""
    delivery_epoch: int = 0

    # Runtime-mutable fields
    messages: list = dataclasses.field(default_factory=list)
    model: str = ""
    turn_index: int = 0
    api_call_count: int = 0
    max_turns: int = 45

    # Branch determinants — populated by begin_run
    provider_name: str = ""
    provider_sdk_version: str = ""
    api_mode: str = "chat"
    streaming_enabled: bool = False
    tool_dispatch_mode: str = "sequential"

    # Per-phase capture
    last_request_hash: Optional[str] = None
    last_response_body: Optional[dict] = None
    last_error_class: Optional[str] = None

    # Connection for Stage B writes (lazy, shared).
    _conn: Optional[sqlite3.Connection] = None


def begin_run(
    agent: Any,
    thread_id: str,
    session_id: str,
    *,
    resume_state: Optional[dict] = None,
) -> StageBRunState:
    """
    Called at the top of run_conversation. Returns StageBRunState with
    .enabled=False if Stage B flag is off (agent-path runs as before).

    If resume_state is provided, state fields are restored from it AND
    resumed=True so the caller can skip re-initialization.
    """
    rs = StageBRunState()
    if not is_enabled():
        return rs

    rs.enabled = True
    rs.thread_id = thread_id
    rs.session_id = session_id
    rs.note = str(getattr(agent, '_durable_note', '') or getattr(agent, '_stage_b_note', '') or 'top_level_run')

    _thread_s = str(thread_id or '')
    _session_s = str(session_id or '')
    if (
        str(getattr(agent, 'platform', '') or '').lower() == 'cron'
        or _thread_s.startswith('cron_')
        or _session_s.startswith('cron_')
        or _thread_s == _session_s
    ):
        rs.enabled = False
        return rs

    if resume_state is not None:
        # Validate before accepting.
        try:
            _ds.verify_resumable(resume_state)
        except _ds.ResumeStateIncompleteError as exc:
            logger.error("stage_b: resume_state incomplete: %s; running as fresh",
                         exc)
            resume_state = None

    if resume_state is not None:
        # Resume path — rehydrate state.
        rs.resumed = True
        rs.run_id = resume_state["run_id"]
        rs.resume_chain_id = resume_state["resume_chain_id"]
        rs.resume_generation = resume_state["resume_generation"]
        rs.agent_profile = resume_state.get("agent_profile", "")
        rs.delivery_epoch = resume_state.get("delivery_epoch", 0)
        rs.messages = list(resume_state.get("messages", []))
        rs.model = resume_state.get("model", "")
        rs.turn_index = resume_state.get("turn_index", 0)
        rs.api_call_count = resume_state.get("api_call_count", 0)
        rs.max_turns = resume_state.get("max_turns", 45)
        rs.provider_name = resume_state.get("provider_name", "")
        rs.provider_sdk_version = resume_state.get("provider_sdk_version", "")
        rs.api_mode = resume_state.get("api_mode", "chat")
        rs.streaming_enabled = resume_state.get("streaming_enabled", False)
        rs.tool_dispatch_mode = resume_state.get("tool_dispatch_mode", "sequential")
        rs.last_request_hash = resume_state.get("last_request_hash")
        rs.last_response_body = resume_state.get("last_response_body")
        logger.info("stage_b: resumed run_id=%s chain=%s gen=%d turn=%d",
                    rs.run_id, rs.resume_chain_id, rs.resume_generation,
                    rs.turn_index)
    else:
        # Fresh run.
        rs.run_id = uuid.uuid4().hex
        rs.resume_chain_id = rs.run_id  # first run's id IS the chain id
        rs.resume_generation = 0
        rs.agent_profile = _infer_agent_profile(agent)
        rs.delivery_epoch = 0
        # Other fields populated by caller as state progresses.

        # Stage B ↔ Stage A: back-fill resume_chain_id into the agent_threads
        # row created by on_agent_start. Without this, scan_and_claim on
        # restart cannot link Stage A's in-flight row to Stage B's
        # checkpoints, so resume returns 0 every boot.
        try:
            _conn = _conn_for_rs(rs)
            _cutoff = int(time.time()) - 60  # last 60s only
            _expected_note = getattr(rs, "note", None) or "top_level_run"
            _expected_session_key = str(thread_id or "")
            _expected_thread_id = str(session_id or thread_id or "")
            _stage_a_run_id = str(getattr(agent, "_durable_stage_a_run_id", "") or "")

            _cur = None
            if _stage_a_run_id:
                _cur = _conn.execute(
                    "UPDATE agent_threads "
                    "SET resume_chain_id = ?, resume_generation = 0 "
                    "WHERE run_id = ? AND terminal_state IS NULL",
                    (rs.resume_chain_id, _stage_a_run_id),
                )

            if _cur is None or _cur.rowcount == 0:
                _cur = _conn.execute(
                    "UPDATE agent_threads "
                    "SET resume_chain_id = ?, resume_generation = 0 "
                    "WHERE rowid = ("
                    "  SELECT rowid FROM agent_threads "
                    "  WHERE terminal_state IS NULL "
                    "    AND resume_chain_id IS NULL "
                    "    AND started_at >= ? "
                    "    AND COALESCE(note, '') = ? "
                    "    AND ((? != '' AND COALESCE(session_key, '') = ?) "
                    "         OR (? != '' AND COALESCE(thread_id, '') = ?)) "
                    "  ORDER BY started_at DESC LIMIT 1"
                    ")",
                    (
                        rs.resume_chain_id,
                        _cutoff,
                        _expected_note,
                        _expected_session_key,
                        _expected_session_key,
                        _expected_thread_id,
                        _expected_thread_id,
                    ),
                )

            _conn.commit()
            if _cur.rowcount > 0:
                logger.info(
                    "stage_b: linked chain=%s to agent_threads "
                    "(thread=%s, session=%s)",
                    rs.resume_chain_id, thread_id, session_id,
                )
            else:
                _now = int(time.time())
                _conn.execute(
                    "INSERT OR IGNORE INTO agent_threads "
                    "(run_id, thread_id, session_key, agent_profile, source, "
                    " started_at, resume_chain_id, resume_generation, note) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)",
                    (
                        rs.run_id,
                        session_id or thread_id,
                        session_id or thread_id,
                        rs.agent_profile,
                        "stage_b_fallback",
                        _now,
                        rs.resume_chain_id,
                        "created by stage_b fallback; Stage A row missing",
                    ),
                )
                _conn.commit()
                logger.warning(
                    "stage_b: created fallback agent_threads row for chain=%s "
                    "(thread=%r session=%r); Stage A on_agent_start row was missing",
                    rs.resume_chain_id, thread_id, session_id,
                )
        except Exception as _exc:
            logger.error("stage_b: chain_id back-fill failed: %s", _exc)

    # Install the tool-dispatcher wrap on this agent instance. This
    # monkey-patches self._execute_tool_calls to interpose pending_commits
    # bookkeeping around the original dispatcher — keeping the tool-
    # wrapping logic OUT of run_agent.py's 300-line dispatcher methods.
    if rs.enabled:
        _install_tool_dispatcher_wrap(agent, rs)

    return rs


def _install_tool_dispatcher_wrap(agent: Any, rs: StageBRunState) -> None:
    """
    Wrap agent._execute_tool_calls with a Stage-B-aware version that:
      1. Computes logical_action_id for each tool_call in the batch.
      2. Writes pending_commits rows for @durable_write tools BEFORE the
         original dispatcher runs (durable-before-effect).
      3. Calls the original dispatcher to actually execute tools.
      4. Inspects the appended tool-result messages to mark each tool
         committed or failed.
      5. For tools with prior committed rows (replay case), skips the
         dispatch entirely.

    Monkey-patches per-instance (setattr on self). Original method restored
    at end_run.
    """
    # Defensive: skip wrap when agent is missing the expected dispatcher
    # attribute. Safe for test fixtures that pass agent=None and for any
    # adapter that invokes begin_run before the AIAgent machinery is fully
    # set up.
    if agent is None or not hasattr(agent, "_execute_tool_calls"):
        return
    if getattr(agent, "_sb_tool_wrap_installed", False):
        return  # already wrapped

    original_execute = agent._execute_tool_calls
    agent._sb_original_execute_tool_calls = original_execute
    agent._sb_tool_wrap_installed = True
    agent._sb_current_rs = rs

    def wrapped_execute_tool_calls(assistant_message, messages, effective_task_id,
                                    api_call_count=0):
        return _execute_tool_calls_durable(
            agent, rs, original_execute,
            assistant_message, messages, effective_task_id, api_call_count,
        )

    agent._execute_tool_calls = wrapped_execute_tool_calls


def _execute_tool_calls_durable(
    agent: Any,
    rs: StageBRunState,
    original_execute: Callable,
    assistant_message: Any,
    messages: list,
    effective_task_id: str,
    api_call_count: int = 0,
) -> None:
    """
    Stage B-aware wrapper around the original _execute_tool_calls.

    Strategy:
      BEFORE original dispatch:
        - For each tool_call in the batch, compute logical_action_id.
        - Check pending_commits:
          * committed → inject cached result into messages, DROP this
            call from the batch so original dispatcher skips it.
          * pending → apply reconciliation (queryable probe / degraded
            escalate / at_most_once mark_failed).
          * no row → write pending row.

      CALL original dispatch on remaining (not-already-committed) calls.
      The dispatcher appends tool-result messages directly to `messages`.

      AFTER original dispatch:
        - For each tool whose result was appended, mark committed with
          the result content.

    Fail-safe: any error in the Stage B path is logged and we fall back
    to calling the original dispatcher unchanged.
    """
    try:
        _load_deps()
    except Exception as exc:
        logger.warning("stage_b: tool wrap deps failed: %s; falling back", exc)
        return original_execute(assistant_message, messages, effective_task_id,
                                api_call_count)

    # Guard: Stage B not enabled for this run.
    if not rs.enabled:
        return original_execute(assistant_message, messages, effective_task_id,
                                api_call_count)

    # Shortcut: if there are no tool_calls, nothing to do.
    tool_calls = getattr(assistant_message, "tool_calls", None) or []
    if not tool_calls:
        return original_execute(assistant_message, messages, effective_task_id,
                                api_call_count)

    # Build per-tool metadata + replay any committed rows.
    conn = _conn_for_rs(rs)
    store = _idem.IdempotencyStore(conn)

    # Track which tools to skip (already committed — replayed) and their
    # cached results, which we splice into messages manually.
    tool_meta: list[dict] = []  # one per tool_call in batch
    for tc in tool_calls:
        try:
            fn_name = _tc_name(tc)
            fn_args = _tc_args(tc)
            call_id = _tc_id(tc)
            lai = _idem.compute_logical_action_id(
                thread_id=rs.thread_id,
                node_name="execute_tools",
                call_id=call_id,
                tool_name=fn_name,
                args_hash=_idem.hash_args(fn_args),
                delivery_epoch=rs.delivery_epoch,
            )
            classification = _idem.get_classification_by_tool_name(fn_name)
            cls_kind = (classification.get("class") if classification else None) or "unclassified"
            mode = (classification.get("reconciliation") if classification else None) or "at_most_once"

            existing = store.get(lai)
            entry = {
                "tc": tc, "lai": lai, "fn_name": fn_name, "call_id": call_id,
                "class": cls_kind, "mode": mode, "existing": existing,
                "replayed": False, "skipped_reason": None,
            }

            # READ tools bypass the commit log entirely.
            if cls_kind == "read":
                tool_meta.append(entry)
                continue

            # Replay: prior committed row → splice cached result into messages.
            if existing is not None and existing.status == "committed":
                cached = (json.loads(existing.result_json)
                          if existing.result_json else None)
                messages.append({
                    "role": "tool",
                    "content": json.dumps(cached, default=str) if not isinstance(cached, str)
                               else cached,
                    "tool_call_id": call_id,
                    "name": fn_name,
                })
                entry["replayed"] = True
                entry["skipped_reason"] = "replayed_committed"
                tool_meta.append(entry)
                continue

            # Orphaned pending — apply per-class reconciliation.
            if existing is not None and existing.status == "pending":
                resolved = _resolve_orphaned_pending(rs, {
                    "call_id": call_id, "tool_name": fn_name,
                    "args": fn_args,
                }, existing, invoke=lambda _tc: None)
                if resolved is not None:
                    messages.append({
                        "role": "tool",
                        "content": json.dumps(resolved, default=str),
                        "tool_call_id": call_id,
                        "name": fn_name,
                    })
                    entry["replayed"] = True
                    entry["skipped_reason"] = "reconciled_orphan"
                    tool_meta.append(entry)
                    continue

            # Fresh write: insert pending row BEFORE dispatcher runs.
            with _transaction(conn):
                store.begin_pending(_idem.PendingCommit(
                    logical_action_id=lai, thread_id=rs.thread_id,
                    run_id=rs.run_id, node_name="execute_tools",
                    call_id=call_id, tool_name=fn_name,
                    payload_hash=_idem.hash_payload(fn_args),
                    status="pending", reconciliation_mode=mode,
                    resume_chain_id=rs.resume_chain_id,
                    resume_generation=rs.resume_generation,
                ))
            tool_meta.append(entry)
        except Exception as exc:
            logger.warning("stage_b: tool-wrap per-call prep failed "
                           "(tool=%s): %s", _tc_name(tc), exc)
            # Fail-safe: mark this call as "no instrumentation" so the
            # dispatcher runs it normally without our interference.
            tool_meta.append({
                "tc": tc, "lai": None, "replayed": False,
                "skipped_reason": f"prep_error: {exc}",
            })

    # If we replayed ANY tool, we need to dispatch only the NON-replayed
    # tools. Build a filtered assistant_message.
    non_replayed_calls = [
        m["tc"] for m in tool_meta if not m["replayed"]
    ]
    if not non_replayed_calls:
        # All tools were replayed — nothing for the dispatcher to do.
        return None

    if len(non_replayed_calls) == len(tool_calls):
        # No tools replayed — run original dispatcher on full batch.
        msg_len_before = len(messages)
        result = original_execute(assistant_message, messages,
                                  effective_task_id, api_call_count)
    else:
        # Partial replay: swap out assistant_message.tool_calls to only
        # include non-replayed ones. Use a shallow proxy so we don't
        # mutate the original object.
        proxy = _AssistantMessageProxy(assistant_message, non_replayed_calls)
        msg_len_before = len(messages)
        result = original_execute(proxy, messages, effective_task_id, api_call_count)

    # AFTER dispatch: mark committed for tools whose results landed in messages.
    _mark_committed_from_messages(rs, store, tool_meta, messages, msg_len_before)
    return result


def _tc_name(tc: Any) -> str:
    """Extract tool_name from a tool_call object (OpenAI SDK shape)."""
    if hasattr(tc, "function") and hasattr(tc.function, "name"):
        return tc.function.name
    if isinstance(tc, dict):
        fn = tc.get("function", {})
        if isinstance(fn, dict):
            return fn.get("name", "?")
    return str(getattr(tc, "name", "?"))


def _tc_args(tc: Any) -> dict:
    """Extract args dict from tool_call (may be JSON string in SDK shape)."""
    if hasattr(tc, "function") and hasattr(tc.function, "arguments"):
        raw = tc.function.arguments
    elif isinstance(tc, dict):
        raw = tc.get("function", {}).get("arguments", {})
    else:
        raw = {}
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"_raw": raw}
    return raw or {}


def _tc_id(tc: Any) -> str:
    """Extract id from tool_call."""
    tid = getattr(tc, "id", None)
    if tid:
        return str(tid)
    if isinstance(tc, dict):
        return str(tc.get("id", f"call_{uuid.uuid4().hex[:8]}"))
    return f"call_{uuid.uuid4().hex[:8]}"


class _AssistantMessageProxy:
    """
    Thin wrapper over an assistant_message that overrides tool_calls.
    Used to pass a filtered batch to the original dispatcher without
    mutating the original message.
    """
    def __init__(self, original: Any, filtered_tool_calls: list):
        self._original = original
        self.tool_calls = filtered_tool_calls

    def __getattr__(self, name):
        return getattr(self._original, name)


def _mark_committed_from_messages(rs: StageBRunState,
                                   store: Any,
                                   tool_meta: list[dict],
                                   messages: list,
                                   msg_len_before: int) -> None:
    """
    After the original dispatcher ran, walk the newly-appended messages
    (messages[msg_len_before:]) and mark each corresponding tool commit
    based on its result message.
    """
    # Map: tool_call_id → result content
    new_messages = messages[msg_len_before:]
    result_by_call_id: dict[str, Any] = {}
    for msg in new_messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "tool":
            continue
        tcid = msg.get("tool_call_id")
        if tcid:
            result_by_call_id[tcid] = msg.get("content")

    for entry in tool_meta:
        if entry.get("replayed"):
            continue
        if entry.get("lai") is None:
            # Prep-error tool — skip commit tracking.
            continue
        if entry.get("class") == "read":
            # Reads have no commit row.
            continue

        lai = entry["lai"]
        call_id = entry["call_id"]
        result = result_by_call_id.get(call_id)
        if result is None:
            # Dispatcher never produced a result for this call — mark failed.
            try:
                with _transaction(store.conn):
                    store.mark_failed(lai, error_message="dispatcher produced no result")
            except Exception as exc:
                logger.warning("stage_b: mark_failed (no-result) for %s: %s",
                               call_id, exc)
            continue

        try:
            # Parse result if JSON.
            if isinstance(result, str):
                try:
                    result_dict = json.loads(result)
                except json.JSONDecodeError:
                    result_dict = {"_raw": result}
            else:
                result_dict = result
            with _transaction(store.conn):
                store.mark_committed(
                    lai,
                    external_response_hash=_idem.hash_payload(result_dict),
                    result_json=json.dumps(result_dict, default=str) if not isinstance(result_dict, str)
                                else result_dict,
                )
        except Exception as exc:
            logger.warning("stage_b: mark_committed for %s failed: %s",
                           call_id, exc)


def _infer_agent_profile(agent: Any) -> str:
    """Read client_identity from config, falling back to HERMES_PROFILE."""
    cfg = _cfg()
    return (cfg.get("client_identity")
            or cfg.get("profile")
            or __import__("os").environ.get("HERMES_PROFILE", "unknown"))


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint write (spec v0.5 §4.3, §4.7)
# ─────────────────────────────────────────────────────────────────────────────

# Phase mapping per spec §4.6.
PHASE_PRE_CALL      = 1
PHASE_POST_RESPONSE = 2
PHASE_POST_TOOLS    = 3


def checkpoint(
    agent: Any,
    rs: StageBRunState,
    *,
    phase: int,
    response_body: Optional[dict] = None,
    checkpoint_seq: int = 0,
) -> None:
    """
    Write a turn checkpoint row. No-op if Stage B disabled.

    Called at three points per turn: phase=PHASE_PRE_CALL at the top,
    phase=PHASE_POST_RESPONSE after response validated (with response_body),
    phase=PHASE_POST_TOOLS after tool dispatch completes.

    Updates rs.last_response_body when phase=POST_RESPONSE so it survives
    any replay. Uses agent.messages (external) as the authoritative
    messages list via a stamp-into-state step.
    """
    if not rs.enabled:
        return

    # Stamp current state snapshot.
    if response_body is not None:
        rs.last_response_body = response_body
    # Caller updates rs.messages etc in-place as turn progresses.

    try:
        conn = _conn_for_rs(rs)
        state_dict = _rs_to_state_dict(rs)
        json_text, digest = _ds.serialize(state_dict)

        # Write agent_turns row. Fencing trigger will abort if another
        # higher-generation writer exists (I2 invariant).
        conn.execute(
            "INSERT OR REPLACE INTO agent_turns "
            "(resume_chain_id, turn_index, phase_ordinal, checkpoint_seq, "
            " resume_generation, state_schema_version, state_parser_version, "
            " state_json, last_request_hash, last_response_body, "
            " state_json_sha256, written_by_run_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                rs.resume_chain_id, rs.turn_index, phase, checkpoint_seq,
                rs.resume_generation,
                _ds.CURRENT_STATE_SCHEMA_VERSION,
                _ds.CURRENT_STATE_PARSER_VERSION,
                json_text,
                rs.last_request_hash,
                json.dumps(response_body, default=str) if response_body else None,
                digest,
                rs.run_id,
                int(time.time()),
            ),
        )
        conn.commit()
    except sqlite3.IntegrityError as exc:
        # Fencing trigger fired — this writer is stale. Log and abort the run.
        logger.error("stage_b: checkpoint REJECTED by fencing for run %s gen %d: %s",
                     rs.run_id, rs.resume_generation, exc)
        # Signal to caller that state is invalid. Raise so the outer loop
        # short-circuits rather than continuing on stale state.
        raise StaleRunAbortError(
            f"run {rs.run_id} is stale (fencing rejected checkpoint write)"
        )
    except sqlite3.OperationalError as exc:
        # DB unavailable. Per spec v0.5 §4.9: abandon the run rather than
        # continue on partial state.
        logger.error("stage_b: checkpoint DB error for run %s: %s; abandoning",
                     rs.run_id, exc)
        _mark_run_abandoned(rs, reason=f"checkpoint_db_error: {exc}")
        raise StageBCheckpointError(f"checkpoint write failed: {exc}")


class StaleRunAbortError(RuntimeError):
    """This run's generation is stale; outer loop must abort immediately."""


class StageBCheckpointError(RuntimeError):
    """Checkpoint write failed; run marked abandoned."""


# ─────────────────────────────────────────────────────────────────────────────
# Tool dispatch wrapping (spec v0.5 §4.7)
# ─────────────────────────────────────────────────────────────────────────────

def dispatch_durable(
    agent: Any,
    rs: StageBRunState,
    tool_call: dict,
    invoke: Callable[[dict], dict],
) -> dict:
    """
    Wrap a single tool-call dispatch with the Stage B two-transaction
    protocol:

      tx1: INSERT pending_commits (durable before external side effect)
      invoke(tool_call)   — runs OUTSIDE any transaction, external side effect
      tx2: UPDATE pending_commits status='committed' with result

    When flag is off: calls invoke() directly and returns its result.

    On orphaned pending rows (e.g., resume after crash mid-dispatch),
    applies per-class reconciliation (queryable/degraded/at_most_once).
    """
    if not rs.enabled:
        return invoke(tool_call)

    _load_deps()

    # Compute logical_action_id — stable across run_id rotation.
    lai = _idem.compute_logical_action_id(
        thread_id=rs.thread_id,
        node_name="execute_tools",
        call_id=tool_call.get("call_id", tool_call.get("id", "?")),
        tool_name=tool_call.get("tool_name", tool_call.get("name", "?")),
        args_hash=_idem.hash_args(tool_call.get("args", tool_call.get("arguments", {}))),
        delivery_epoch=rs.delivery_epoch,
    )
    tool_call["logical_action_id"] = lai

    conn = _conn_for_rs(rs)
    store = _idem.IdempotencyStore(conn)
    existing = store.get(lai)

    # Fast path: prior committed result.
    if existing is not None and existing.status == "committed":
        logger.info("stage_b: replay committed result for tool %s",
                    tool_call.get("tool_name"))
        if existing.result_json:
            return json.loads(existing.result_json)
        return {"ok": True, "replayed": True}

    # Orphaned pending — delegate to per-class reconciliation.
    if existing is not None and existing.status == "pending":
        return _resolve_orphaned_pending(rs, tool_call, existing, invoke)

    # Classify tool (@durable_read bypasses commit log).
    classification = _idem.get_classification_by_tool_name(
        tool_call.get("tool_name", tool_call.get("name", "?"))
    )
    if classification and classification["class"] == "read":
        return invoke(tool_call)

    # @durable_write / unclassified: full tx1 + invoke + tx2.
    mode = (classification.get("reconciliation") if classification else None) \
           or "at_most_once"

    # tx1
    commit = _idem.PendingCommit(
        logical_action_id=lai,
        thread_id=rs.thread_id,
        run_id=rs.run_id,
        node_name="execute_tools",
        call_id=tool_call.get("call_id", tool_call.get("id", "?")),
        tool_name=tool_call.get("tool_name", tool_call.get("name", "?")),
        payload_hash=_idem.hash_payload(tool_call.get("args", {})),
        status="pending",
        reconciliation_mode=mode,
        resume_chain_id=rs.resume_chain_id,
        resume_generation=rs.resume_generation,
    )
    store.begin_pending(commit)
    conn.commit()

    # Handler runs outside the transaction.
    try:
        result = invoke(tool_call)
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        with _transaction(conn):
            store.mark_failed(lai, error_message=error_msg)
        raise

    # tx2
    with _transaction(conn):
        store.mark_committed(
            lai,
            external_response_hash=_idem.hash_payload(result),
            result_json=json.dumps(result, default=str),
        )
    return result


def _resolve_orphaned_pending(
    rs: StageBRunState,
    tool_call: dict,
    existing: Any,
    invoke: Callable[[dict], dict],
) -> dict:
    """Per-class reconciliation for orphaned pending_commits row."""
    conn = _conn_for_rs(rs)
    store = _idem.IdempotencyStore(conn)
    lai = existing.logical_action_id
    mode = existing.reconciliation_mode

    if mode == "at_most_once":
        # Never re-execute; mark failed, return synthetic error.
        with _transaction(conn):
            store.mark_failed(lai, error_message="orphan_at_most_once")
        return {
            "ok": False,
            "error": "orphan at_most_once — tool not re-executed on resume",
            "tool_name": tool_call.get("tool_name"),
        }

    if mode == "degraded":
        # Escalate to operator; never reissue.
        with _transaction(conn):
            store.mark_failed(lai, error_message="orphan_degraded")
        logger.error("stage_b: orphan degraded tool %s — operator resolution required",
                     tool_call.get("tool_name"))
        return {
            "ok": False,
            "error": "orphan degraded — operator resolution required",
            "tool_name": tool_call.get("tool_name"),
        }

    if mode == "queryable":
        # Queryable reconciliation: probe external, inject payload on committed.
        # For Phase 1: no reconciler registry wired in here — fall through
        # to "escalate for operator" until a reconciler interface is stood
        # up for a specific tool. This is the safe default per spec v0.5 §2.
        with _transaction(conn):
            store.mark_failed(lai, error_message="orphan_queryable_no_reconciler")
        return {
            "ok": False,
            "error": "orphan queryable — reconciler not yet wired",
            "tool_name": tool_call.get("tool_name"),
        }

    # Unknown mode — fail-safe
    with _transaction(conn):
        store.mark_failed(lai, error_message=f"orphan_unknown_mode_{mode}")
    return {"ok": False, "error": f"unknown reconciliation mode {mode}"}


# ─────────────────────────────────────────────────────────────────────────────
# Run termination
# ─────────────────────────────────────────────────────────────────────────────

def end_run(
    agent: Any,
    rs: StageBRunState,
    *,
    terminal_state: str,
    note: Optional[str] = None,
) -> None:
    """
    Called at run completion. Updates agent_threads terminal_state with
    generation-fenced WHERE clause. Stage A's on_agent_end handles the
    shared terminal tracking; this just ensures Stage B's version of the
    generation column is respected.

    Called even when rs.enabled=False (no-op path).
    """
    if not rs.enabled:
        return

    try:
        conn = _conn_for_rs(rs)
        # Fenced UPDATE — only applies if our generation is current.
        # Stage B's run_id is a different UUID from Stage A's row.run_id, so
        # match by resume_chain_id (which we back-filled at begin_run) +
        # generation fence.
        cur = conn.execute(
            "UPDATE agent_threads "
            "SET terminal_state = ?, ended_at = ?, note = COALESCE(?, note) "
            "WHERE resume_chain_id = ? AND resume_generation = ?",
            (terminal_state, int(time.time()), note,
             rs.resume_chain_id, rs.resume_generation),
        )
        conn.commit()
        if cur.rowcount == 0:
            logger.warning(
                "stage_b: end_run no-op for %s gen %d (already transitioned or stale)",
                rs.run_id, rs.resume_generation,
            )
    except Exception as exc:
        logger.error("stage_b: end_run failed for %s: %s", rs.run_id, exc)

    # Stage C: promise extraction. Only on SUCCESS — failed/abandoned runs
    # didn't get to deliver a final response, so no commitments to extract.
    if terminal_state == "SUCCESS":
        try:
            _maybe_extract_promises(agent, rs)
        except Exception as exc:
            # Stage C is best-effort; never block end_run on extraction failures.
            logger.warning("stage_c: promise extraction failed for run %s: %s",
                           rs.run_id, exc)




def _maybe_extract_promises(agent, rs) -> None:
    """Run Stage C extraction if flag is on. Best-effort, never raises."""
    cfg = _cfg()
    if not _is_stage_c_enabled(cfg):
        return

    try:
        try:
            from . import stage_c_promises as _sc
        except ImportError:
            import stage_c_promises as _sc  # type: ignore[no-redef]
    except ImportError as exc:
        logger.debug("stage_c: module not importable; skipping (%s)", exc)
        return

    conn = _conn_for_rs(rs)
    _sc.initialize_schema(conn)

    extraction_fn = _build_extraction_fn(cfg)
    if extraction_fn is None:
        logger.debug("stage_c: no extraction_fn available; skipping")
        return

    promises = _sc.extract_promises(
        messages=rs.messages,
        extraction_fn=extraction_fn,
    )
    if not promises:
        return

    for pdata in promises:
        pid = _sc.Promise.gen_id(
            rs.resume_chain_id,
            pdata["promise_text"],
            int(time.time()),
        )
        promise = _sc.Promise(
            promise_id=pid,
            resume_chain_id=rs.resume_chain_id,
            thread_id=rs.thread_id,
            agent_profile=rs.agent_profile,
            promise_text=pdata["promise_text"],
            trigger_type=pdata["trigger_type"],
            trigger_condition=pdata["trigger_condition"],
            due_at=pdata.get("due_at"),
            made_at=int(time.time()),
            made_by_run_id=rs.run_id,
            extraction_source="llm_extraction",
        )
        _sc.record_promise(conn, promise)
    logger.info("stage_c: extracted %d promise(s) from run %s",
                len(promises), rs.run_id)


def _is_stage_c_enabled(cfg: dict) -> bool:
    """Stage C requires Stage B (which requires Stage A)."""
    if not cfg.get("durable_runtime_stage_b"):
        return False
    raw = cfg.get("durable_runtime_stage_c", False)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in ("true", "yes", "1", "on")
    return bool(raw)


def _build_extraction_fn(cfg: dict):
    """Build extraction_fn(prompt) -> str using auxiliary.promise_extraction
    (or fallback aux configs). Returns None if unavailable."""
    import os as _os, json as _json
    aux = cfg.get("auxiliary", {}) if isinstance(cfg, dict) else {}
    extraction_cfg = (aux.get("promise_extraction") or
                      aux.get("session_search") or
                      aux.get("compression") or
                      aux.get("approval") or {})
    if not isinstance(extraction_cfg, dict):
        return None
    model = extraction_cfg.get("model")
    base_url = extraction_cfg.get("base_url") or cfg.get("model", {}).get("base_url")
    if not model or not base_url:
        return None

    api_key_env = (extraction_cfg.get("api_key_env")
                   or cfg.get("model", {}).get("api_key_env")
                   or "OPENROUTER_API_KEY")
    api_key = _os.environ.get(api_key_env, "")
    if not api_key:
        env_file = Path.home() / ".hermes" / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith(api_key_env + "="):
                    api_key = line.split("=", 1)[1].strip().strip("'\"")
                    break
    if not api_key:
        return None

    timeout_s = float(extraction_cfg.get("timeout", 30))
    base_url = base_url.rstrip("/")

    def call(prompt: str) -> str:
        import urllib.request, urllib.error
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=_json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 1024,
            }).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = _json.loads(resp.read().decode("utf-8", errors="replace"))
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
            logger.warning("stage_c: extraction API call failed: %s", exc)
            return ""
        try:
            return body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            return ""

    return call


def _mark_run_abandoned(rs: StageBRunState, *, reason: str) -> None:
    """Emergency path — called when checkpoint DB fails mid-run."""
    try:
        conn = _conn_for_rs(rs)
        conn.execute(
            "UPDATE agent_threads "
            "SET terminal_state = 'ABANDONED', ended_at = ?, "
            "    note = COALESCE(note,'') || ' | ' || ? "
            "WHERE run_id = ?",
            (int(time.time()), reason, rs.run_id),
        )
        conn.commit()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _conn_for_rs(rs: StageBRunState) -> sqlite3.Connection:
    """Lazy-open the durable DB connection; cache on the rs."""
    if rs._conn is None:
        _load_deps()
        rs._conn = _di._open_db()  # reuses Stage A's open path
    return rs._conn


def _transaction(conn: sqlite3.Connection):
    """
    Explicit transaction context manager. Necessary because the durable
    DB connection is opened in autocommit mode (isolation_level=None).
    """
    class _Txn:
        def __enter__(self_):
            conn.execute("BEGIN IMMEDIATE")
            return conn

        def __exit__(self_, exc_type, exc, tb):
            if exc_type is None:
                conn.execute("COMMIT")
            else:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.OperationalError:
                    pass
            return False  # don't suppress
    return _Txn()


def _rs_to_state_dict(rs: StageBRunState) -> dict:
    """Convert StageBRunState to the StageBState TypedDict form."""
    _load_deps()
    return {
        "state_schema_version": _ds.CURRENT_STATE_SCHEMA_VERSION,
        "state_parser_version": _ds.CURRENT_STATE_PARSER_VERSION,
        "resume_chain_id": rs.resume_chain_id,
        "resume_generation": rs.resume_generation,
        "run_id": rs.run_id,
        "thread_id": rs.thread_id,
        "session_id": rs.session_id,
        "agent_profile": rs.agent_profile,
        "delivery_epoch": rs.delivery_epoch,
        "messages": list(rs.messages),
        "model": rs.model,
        "turn_index": rs.turn_index,
        "max_turns": rs.max_turns,
        "api_call_count": rs.api_call_count,
        "provider_name": rs.provider_name,
        "provider_sdk_version": rs.provider_sdk_version,
        "api_mode": rs.api_mode,
        "streaming_enabled": rs.streaming_enabled,
        "tool_dispatch_mode": rs.tool_dispatch_mode,
        "retry_state": {},
        "approval_state": None,
        "compression_state": None,
        "code_version": "stage-b-phase-1",
        "pending_tool_calls": [],
        "completed_tool_calls": [],
        "last_error_class": rs.last_error_class,
        "last_request_hash": rs.last_request_hash,
        "last_response_body": rs.last_response_body,
        "started_at": int(time.time()),
        "last_checkpoint_at": int(time.time()),
    }
