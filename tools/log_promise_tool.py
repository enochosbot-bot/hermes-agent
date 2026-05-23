"""
tools/log_promise_tool.py — explicit promise logging for the agent.

When an agent commits to future work it SHOULD call this tool to record
the commitment durably. The agent's SOUL.md should encourage this over
implicit "I'll check back later" statements that rely on LLM extraction.

Tool name: log_promise
Toolset:   durability

Args:
  promise_text         (str)  — the commitment, paraphrased or verbatim
  trigger_type         (str)  — 'time' | 'user_action' | 'agent_condition'
  trigger_condition    (str)  — concise description of when the promise fires
  due_at_iso           (str)  — optional ISO datetime if time-based

Returns: { ok: bool, promise_id: str | null, error: str | null }
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger("tools.log_promise")


# Schema for tool registry — discoverable via tools.registry
LOG_PROMISE_SCHEMA = {
    "name": "log_promise",
    "description": (
        "Durably record a commitment to future action. Call this whenever "
        "you say you will do something later — checking back, sending an "
        "update, following up after an event. Logged promises are tracked "
        "by the durable runtime; if you forget, the system will re-engage "
        "you when the trigger fires. NEVER rely on memory alone for "
        "future commitments."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "promise_text": {
                "type": "string",
                "description": "Concise statement of what you committed to do."
            },
            "trigger_type": {
                "type": "string",
                "enum": ["time", "user_action", "agent_condition"],
                "description": (
                    "What kind of trigger fires this promise:\n"
                    "  time            — at a specific datetime / after a duration\n"
                    "  user_action     — waiting for the user to do or say something\n"
                    "  agent_condition — tied to an external condition you'll check"
                ),
            },
            "trigger_condition": {
                "type": "string",
                "description": (
                    "Concise description of when the promise fires. For "
                    "time-based: the time descriptor (e.g. 'tomorrow 9am'). "
                    "For user_action: what the user must do. For "
                    "agent_condition: the check you'll perform."
                ),
            },
            "due_at_iso": {
                "type": "string",
                "description": (
                    "Optional ISO 8601 UTC datetime if time-based and "
                    "specific. Required for sweeper-driven re-engagement."
                ),
            },
        },
        "required": ["promise_text", "trigger_type", "trigger_condition"],
    },
}


def log_promise_tool(args: dict, **kwargs: Any) -> str:
    """
    Tool handler. Returns a JSON string with {ok, promise_id, error}.

    Reads runtime context from kwargs / globals to identify the current
    chain/thread/run. The runtime injects these via the agent's
    self._sb_current_rs that we set in stage_b_hooks._install_tool_dispatcher_wrap.
    """
    try:
        from agent.idempotency import durable_write
    except ImportError:
        durable_write = None  # tool still works; classification is metadata

    promise_text = str(args.get("promise_text", "")).strip()
    trigger_type = str(args.get("trigger_type", "")).strip()
    trigger_condition = str(args.get("trigger_condition", "")).strip()
    due_at_iso = args.get("due_at_iso")

    if not promise_text:
        return _err("promise_text is required")
    if trigger_type not in ("time", "user_action", "agent_condition"):
        return _err("trigger_type must be one of: time, user_action, agent_condition")
    if not trigger_condition:
        return _err("trigger_condition is required")

    # Parse due_at if provided.
    due_at = None
    if due_at_iso:
        try:
            parsed = _dt.datetime.fromisoformat(
                str(due_at_iso).replace("Z", "+00:00")
            )
            due_at = int(parsed.timestamp())
        except (ValueError, TypeError) as exc:
            return _err(f"due_at_iso parse failed: {exc}")

    # Find the agent's current StageBRunState. The wrap installer at
    # _install_tool_dispatcher_wrap stashes self._sb_current_rs.
    agent = kwargs.get("agent") or kwargs.get("self")
    rs = getattr(agent, "_sb_current_rs", None) if agent is not None else None
    if rs is None or not getattr(rs, "enabled", False):
        # Stage B not enabled — promise tool can't durably record.
        # Fall back to logging a warning. This makes the tool a soft no-op
        # in non-Stage-B environments rather than an error.
        logger.warning("log_promise: Stage B not enabled; promise NOT recorded "
                       "(text=%r)", promise_text[:80])
        return _ok(promise_id=None, note="stage_b_disabled")

    # Open the durable DB (same path as Stage B uses).
    try:
        try:
            from agent import durable_integration as _di
        except ImportError:
            import durable_integration as _di  # type: ignore[no-redef]
        try:
            from agent import stage_c_promises as _sc
        except ImportError:
            import stage_c_promises as _sc  # type: ignore[no-redef]
    except ImportError as exc:
        return _err(f"durable runtime modules unavailable: {exc}")

    conn = _di._open_db()
    _sc.initialize_schema(conn)

    pid = _sc.log_promise_explicit(
        conn,
        resume_chain_id=rs.resume_chain_id,
        thread_id=rs.thread_id,
        agent_profile=rs.agent_profile,
        run_id=rs.run_id,
        promise_text=promise_text,
        trigger_type=trigger_type,
        trigger_condition=trigger_condition,
        due_at=due_at,
    )
    return _ok(promise_id=pid)


def _ok(*, promise_id: str | None = None, note: str | None = None) -> str:
    return json.dumps({"ok": True, "promise_id": promise_id, "note": note})


def _err(msg: str) -> str:
    return json.dumps({"ok": False, "error": msg, "promise_id": None})


def _check_log_promise_available() -> bool:
    """Always available — failure mode is a recorded warning, not a hard error."""
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Registry registration (auto-discovered by tools.registry — must be a
# top-level `registry.register(...)` Expr for AST-based discovery.)
# ─────────────────────────────────────────────────────────────────────────────

from tools.registry import registry

try:
    from agent.idempotency import register_durability
    register_durability("log_promise", class_="write", reconciliation="queryable")
except Exception:
    pass

registry.register(
    name="log_promise",
    toolset="durability",
    schema=LOG_PROMISE_SCHEMA,
    handler=log_promise_tool,
    check_fn=_check_log_promise_available,
    emoji="📌",
)
