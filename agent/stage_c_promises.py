"""
agent/stage_c_promises.py — commitment durability.

Stage C extends the durable runtime to track agent COMMITMENTS — future
actions the agent said it would take. Closes the semantic gap that
Stage A (per-run tracking) and Stage B (per-turn checkpointing) don't
cover: "I'll check back tomorrow" isn't a run-crash, it's a deferred
obligation that current durability forgets the moment the run ends SUCCESS.

Design:

  1. Promise record  — agent_promises table keyed by promise_id, FK to
     resume_chain_id so promises attach to the conversation lineage.

  2. Extraction       — after every run completes SUCCESS, a lightweight
     auxiliary-model pass reads the last 8 messages and extracts
     structured promise records (0..N per run).

  3. Explicit logging — agent can call `log_promise` tool to explicitly
     record a commitment, bypassing extraction. Recommended when the
     commitment is non-obvious or has specific conditions. SOUL.md should
     encourage this.

  4. Sweeper          — periodic timer queries pending promises. For
     due time-based ones, it enqueues a re-engagement prompt to the
     thread (via existing inbox queue infra). User_action-waiting
     promises fire only when the user posts a follow-up. Agent-condition
     promises are polled via a separate check (or left to operator).

  5. Lifecycle        — pending → fulfilled (agent acts on it) / expired
     (N sweeper retries, no fulfillment) / cancelled (agent or operator).

Safety:
  - Schema is additive; existing Stage A+B data untouched.
  - Feature-flag gated: durable_runtime_stage_c (requires stage_b=True).
  - Extraction is best-effort. Failures are logged; the run still
    terminates normally. Missing a promise is annoying, not dangerous.
  - The sweeper's re-engagement goes through the normal message queue,
    so it participates in Stage A+B tracking (a promise-triggered run
    gets its own run_id + checkpoints).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional

logger = logging.getLogger("agent.stage_c_promises")


# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────

PROMISES_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_promises (
    promise_id          TEXT PRIMARY KEY,
    resume_chain_id     TEXT NOT NULL,          -- conversation lineage
    thread_id           TEXT NOT NULL,
    agent_profile       TEXT NOT NULL,
    promise_text        TEXT NOT NULL,          -- what the agent said
    trigger_type        TEXT NOT NULL
                         CHECK(trigger_type IN ('time','user_action','agent_condition')),
    trigger_condition   TEXT NOT NULL,          -- parsed form: ISO date, description, predicate
    due_at              INTEGER,                -- epoch seconds for time-based; NULL otherwise
    made_at             INTEGER NOT NULL,
    made_by_run_id      TEXT NOT NULL,
    extraction_source   TEXT NOT NULL           -- 'llm_extraction' | 'explicit_tool'
                         CHECK(extraction_source IN ('llm_extraction','explicit_tool')),
    state               TEXT NOT NULL           -- lifecycle
                         CHECK(state IN ('pending','fulfilled','expired','cancelled')),
    sweep_attempts      INTEGER NOT NULL DEFAULT 0,
    last_swept_at       INTEGER,
    fulfilled_by_run_id TEXT,
    fulfilled_at        INTEGER,
    state_changed_at    INTEGER NOT NULL,
    note                TEXT
);
CREATE INDEX IF NOT EXISTS idx_agent_promises_chain
    ON agent_promises(resume_chain_id, state);
CREATE INDEX IF NOT EXISTS idx_agent_promises_due
    ON agent_promises(state, due_at, sweep_attempts)
    WHERE state = 'pending';
CREATE INDEX IF NOT EXISTS idx_agent_promises_thread
    ON agent_promises(thread_id, state, made_at);
"""


MAX_SWEEP_ATTEMPTS = 3              # after this many re-engagements, expire
SWEEP_INTERVAL_S = 900              # 15 minutes (matches launchd plist)
USER_ACTION_TIMEOUT_S = 7 * 86400   # after 7 days of no user reply, user_action
                                    # promises expire


def initialize_schema(conn: sqlite3.Connection) -> None:
    """Idempotent — create agent_promises table + indexes if absent."""
    conn.executescript(PROMISES_SCHEMA)
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# State dataclass
# ─────────────────────────────────────────────────────────────────────────────

TriggerType = Literal["time", "user_action", "agent_condition"]
PromiseState = Literal["pending", "fulfilled", "expired", "cancelled"]
ExtractionSource = Literal["llm_extraction", "explicit_tool"]


@dataclass
class Promise:
    promise_id: str
    resume_chain_id: str
    thread_id: str
    agent_profile: str
    promise_text: str
    trigger_type: TriggerType
    trigger_condition: str
    due_at: Optional[int]
    made_at: int
    made_by_run_id: str
    extraction_source: ExtractionSource
    state: PromiseState = "pending"
    sweep_attempts: int = 0
    last_swept_at: Optional[int] = None
    fulfilled_by_run_id: Optional[str] = None
    fulfilled_at: Optional[int] = None
    state_changed_at: int = field(default_factory=lambda: int(time.time()))
    note: Optional[str] = None

    @staticmethod
    def gen_id(chain_id: str, promise_text: str, made_at: int) -> str:
        """Deterministic promise_id — same (chain, text, time) → same id."""
        h = hashlib.sha256(
            f"{chain_id}|{promise_text}|{made_at}".encode("utf-8")
        ).hexdigest()
        return f"prom_{h[:16]}"


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def record_promise(conn: sqlite3.Connection, p: Promise) -> bool:
    """
    Insert the promise. Returns True on new insert, False if duplicate
    (same promise_id already exists) — idempotent at the ID level.
    """
    try:
        conn.execute(
            """INSERT INTO agent_promises (
                promise_id, resume_chain_id, thread_id, agent_profile,
                promise_text, trigger_type, trigger_condition, due_at,
                made_at, made_by_run_id, extraction_source, state,
                sweep_attempts, last_swept_at, fulfilled_by_run_id,
                fulfilled_at, state_changed_at, note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                p.promise_id, p.resume_chain_id, p.thread_id, p.agent_profile,
                p.promise_text, p.trigger_type, p.trigger_condition, p.due_at,
                p.made_at, p.made_by_run_id, p.extraction_source, p.state,
                p.sweep_attempts, p.last_swept_at, p.fulfilled_by_run_id,
                p.fulfilled_at, p.state_changed_at, p.note,
            ),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # Duplicate promise_id — already recorded.
        return False


def mark_fulfilled(conn: sqlite3.Connection, *, promise_id: str,
                   run_id: str, note: Optional[str] = None) -> bool:
    """Transition pending → fulfilled. Returns True on success (1 row updated)."""
    cur = conn.execute(
        """UPDATE agent_promises
           SET state = 'fulfilled',
               fulfilled_by_run_id = ?,
               fulfilled_at = ?,
               state_changed_at = ?,
               note = COALESCE(?, note)
           WHERE promise_id = ? AND state = 'pending'""",
        (run_id, int(time.time()), int(time.time()), note, promise_id),
    )
    conn.commit()
    return cur.rowcount > 0


def mark_expired(conn: sqlite3.Connection, *, promise_id: str,
                 reason: str) -> bool:
    cur = conn.execute(
        """UPDATE agent_promises
           SET state = 'expired',
               state_changed_at = ?,
               note = COALESCE(note, '') || ' | expired: ' || ?
           WHERE promise_id = ? AND state = 'pending'""",
        (int(time.time()), reason, promise_id),
    )
    conn.commit()
    return cur.rowcount > 0


def mark_cancelled(conn: sqlite3.Connection, *, promise_id: str,
                   reason: str) -> bool:
    cur = conn.execute(
        """UPDATE agent_promises
           SET state = 'cancelled',
               state_changed_at = ?,
               note = COALESCE(note, '') || ' | cancelled: ' || ?
           WHERE promise_id = ? AND state IN ('pending','fulfilled')""",
        (int(time.time()), reason, promise_id),
    )
    conn.commit()
    return cur.rowcount > 0


def list_pending(conn: sqlite3.Connection, thread_id: Optional[str] = None) -> list[dict]:
    """Return all pending promises, optionally scoped to a thread."""
    if thread_id is None:
        sql = ("SELECT * FROM agent_promises WHERE state = 'pending' "
               "ORDER BY made_at DESC")
        args: tuple = ()
    else:
        sql = ("SELECT * FROM agent_promises "
               "WHERE state = 'pending' AND thread_id = ? "
               "ORDER BY made_at DESC")
        args = (thread_id,)
    rows = conn.execute(sql, args).fetchall()
    # Return as dicts keyed by column name.
    col_names = [d[0] for d in (conn.execute(sql, args).description or [])]
    return [dict(zip(col_names, row)) for row in rows]


def find_due_time_based(conn: sqlite3.Connection, *,
                        now: Optional[int] = None) -> list[dict]:
    """
    Return time-based pending promises whose due_at has passed AND which
    haven't exceeded MAX_SWEEP_ATTEMPTS. Sorted by due_at ascending.
    """
    now = now if now is not None else int(time.time())
    cur = conn.execute(
        """SELECT * FROM agent_promises
           WHERE state = 'pending'
             AND trigger_type = 'time'
             AND due_at <= ?
             AND sweep_attempts < ?
           ORDER BY due_at ASC
           LIMIT 100""",
        (now, MAX_SWEEP_ATTEMPTS),
    )
    cols = [d[0] for d in cur.description or []]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def increment_sweep(conn: sqlite3.Connection, promise_id: str) -> None:
    conn.execute(
        """UPDATE agent_promises
           SET sweep_attempts = sweep_attempts + 1,
               last_swept_at = ?
           WHERE promise_id = ?""",
        (int(time.time()), promise_id),
    )
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# LLM extraction
# ─────────────────────────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """\
You scan the final messages of a conversation to extract COMMITMENTS the \
ASSISTANT made to perform future actions.

A commitment is an explicit or strongly-implied promise to do something \
later — e.g. "I'll check back tomorrow", "I'll send you the report after \
the meeting", "Let me research that and get back to you", "I'll ping you \
when the deploy finishes".

NOT commitments:
  - Past actions the assistant did ("I sent the email")
  - Informational statements ("the meeting is at 3pm")
  - Suggestions WITHOUT a first-person commitment ("you should ask about X")
  - Clarifying questions ("do you want option A or B?")
  - Filler phrases without substantive commitment ("hope that helps!", \
    "let me know if you need more")

For each commitment you find, classify:
  trigger_type:
    - "time"            — tied to a specific time/date/duration
    - "user_action"     — waiting for a user to do/say something
    - "agent_condition" — tied to some external condition the agent will check

  trigger_condition: A concise description of the trigger.
    - For time-based: an ISO 8601 datetime (UTC) if inferable, or a relative
      descriptor like "+1 day", "+3 hours", "next Monday morning".
    - For user_action: what the user needs to do ("replies with preference",
      "confirms the approach").
    - For agent_condition: the check the agent will run ("meeting ends",
      "deploy completes").

Return a JSON array of objects:
  [
    {
      "promise_text": "string — the verbatim or paraphrased commitment",
      "trigger_type": "time" | "user_action" | "agent_condition",
      "trigger_condition": "string",
      "due_at_iso":  "ISO datetime if time-based and inferable, else null"
    },
    ...
  ]

Return [] if there are no commitments. Do NOT wrap in markdown fences.
Return raw JSON only.

--- CONVERSATION TAIL (last 8 messages) ---
__CONVERSATION_TAIL__
--- END ---

Now reference date/time: __NOW_ISO__
"""


def extract_promises(
    *,
    messages: list[dict],
    extraction_fn: Callable[[str], str],
    now: Optional[int] = None,
) -> list[dict]:
    """
    Call the auxiliary model to extract structured promises from the
    conversation tail. `extraction_fn(prompt: str) -> str` is an
    injectable callable that returns the raw model reply (stripping any
    markdown fences etc).

    Returns a list of dicts with keys promise_text, trigger_type,
    trigger_condition, due_at (int epoch or None).
    """
    now = now or int(time.time())
    tail = messages[-8:] if len(messages) > 8 else list(messages)
    tail_text = "\n".join(
        f"[{m.get('role', '?')}] {m.get('content', '')[:2000]}"
        for m in tail
    )
    import datetime as _dt
    now_iso = _dt.datetime.utcfromtimestamp(now).isoformat() + "Z"
    prompt = (
        EXTRACTION_PROMPT
        .replace("__CONVERSATION_TAIL__", tail_text)
        .replace("__NOW_ISO__", now_iso)
    )

    raw = extraction_fn(prompt)
    if not raw:
        return []

    raw = raw.strip()
    if raw.startswith("```"):
        # Strip fences
        parts = raw.splitlines()
        if parts:
            parts = parts[1:]
        if parts and parts[-1].strip().startswith("```"):
            parts = parts[:-1]
        raw = "\n".join(parts).strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("stage_c: extraction response not JSON: %s; raw=%r",
                       exc, raw[:200])
        return []

    if not isinstance(parsed, list):
        logger.warning("stage_c: extraction returned non-list: %r", type(parsed))
        return []

    out: list[dict] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        promise_text = str(item.get("promise_text", "")).strip()
        if not promise_text:
            continue
        trigger_type = item.get("trigger_type", "agent_condition")
        if trigger_type not in ("time", "user_action", "agent_condition"):
            trigger_type = "agent_condition"
        trigger_condition = str(item.get("trigger_condition", "")).strip() \
                             or "unspecified"
        due_at = None
        if trigger_type == "time":
            due_iso = item.get("due_at_iso")
            if due_iso:
                try:
                    parsed_dt = _dt.datetime.fromisoformat(
                        str(due_iso).replace("Z", "+00:00")
                    )
                    due_at = int(parsed_dt.timestamp())
                except (ValueError, TypeError):
                    due_at = None
        out.append({
            "promise_text": promise_text,
            "trigger_type": trigger_type,
            "trigger_condition": trigger_condition,
            "due_at": due_at,
        })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Sweeper — called by the periodic timer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SweepReport:
    due_found: int = 0
    reengaged: int = 0
    expired: int = 0
    user_action_expired: int = 0
    errors: int = 0


def sweep_due_promises(
    conn: sqlite3.Connection,
    *,
    reengage_fn: Callable[[dict], bool],
    now: Optional[int] = None,
) -> SweepReport:
    """
    For each due time-based promise, call reengage_fn(promise_dict) which
    enqueues a re-engagement prompt to the thread. Increments
    sweep_attempts. Expires after MAX_SWEEP_ATTEMPTS.

    Also expires user_action promises older than USER_ACTION_TIMEOUT_S.

    `reengage_fn` returns True if re-engagement was enqueued successfully,
    False on failure (promise stays pending, sweep_attempts still
    incremented — prevents stuck-loops).
    """
    report = SweepReport()
    now = now or int(time.time())

    # Time-based promises that are due.
    due = find_due_time_based(conn, now=now)
    report.due_found = len(due)

    for p in due:
        try:
            ok = reengage_fn(p)
            increment_sweep(conn, p["promise_id"])
            if ok:
                report.reengaged += 1
            if p["sweep_attempts"] + 1 >= MAX_SWEEP_ATTEMPTS and not ok:
                # Last attempt failed — expire.
                mark_expired(conn, promise_id=p["promise_id"],
                             reason="max_sweep_attempts_reached")
                report.expired += 1
            elif p["sweep_attempts"] + 1 >= MAX_SWEEP_ATTEMPTS:
                # Hit the retry cap even if re-engagement worked.
                mark_expired(conn, promise_id=p["promise_id"],
                             reason="max_sweep_attempts_reached")
                report.expired += 1
        except Exception as exc:
            logger.error("sweep_due: error on promise %s: %s",
                         p["promise_id"], exc)
            report.errors += 1

    # Expire stale user_action promises (nobody replied for USER_ACTION_TIMEOUT_S).
    cur = conn.execute(
        """SELECT promise_id FROM agent_promises
           WHERE state = 'pending'
             AND trigger_type = 'user_action'
             AND made_at < ?""",
        (now - USER_ACTION_TIMEOUT_S,),
    )
    for row in cur.fetchall():
        if mark_expired(conn, promise_id=row[0], reason="user_action_timeout"):
            report.user_action_expired += 1

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Tool-side API — for the `log_promise` tool and SOUL.md guidance
# ─────────────────────────────────────────────────────────────────────────────

def log_promise_explicit(
    conn: sqlite3.Connection,
    *,
    resume_chain_id: str,
    thread_id: str,
    agent_profile: str,
    run_id: str,
    promise_text: str,
    trigger_type: TriggerType,
    trigger_condition: str,
    due_at: Optional[int] = None,
) -> str:
    """
    Called by the agent via the log_promise tool. Returns the promise_id
    so the agent can reference it in later tool calls (mark_fulfilled).
    """
    now = int(time.time())
    pid = Promise.gen_id(resume_chain_id, promise_text, now)
    p = Promise(
        promise_id=pid,
        resume_chain_id=resume_chain_id,
        thread_id=thread_id,
        agent_profile=agent_profile,
        promise_text=promise_text,
        trigger_type=trigger_type,
        trigger_condition=trigger_condition,
        due_at=due_at,
        made_at=now,
        made_by_run_id=run_id,
        extraction_source="explicit_tool",
    )
    record_promise(conn, p)
    return pid
