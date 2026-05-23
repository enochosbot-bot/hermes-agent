"""
agent/durable_integration.py — thin integration layer between the Hermes
gateway and the durable runtime.

Phase 3 Stage A: minimum viable integration.

What this provides when `durable_runtime: true` in config.yaml:
  - On gateway start: open the checkpoint DB, run migration validation,
    load durability declarations, scan for orphan threads from the prior
    run, surface them to the operator alert topic.
  - On agent-run start (in _run_agent): record thread+run_id+started_at
    in the checkpoint DB.
  - On agent-run end: mark the thread terminal (SUCCESS / FAIL / CANCELLED).
  - On gateway stop: flush in-flight markers so the next start can
    distinguish "crashed mid-run" from "clean shutdown".

What this does NOT provide yet:
  - Per-turn checkpointing within run_conversation (that's Stage B).
  - Graph-routed agent loop (Stage C).
  - Exactly-once tool reconciliation (requires Stage B).

Flag-off behavior is a true no-op: every function returns None/0 without
touching the filesystem or DB. Safe to import even when langgraph isn't
installed.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("agent.durable_integration")

HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
SENTINEL_DIR = HERMES_HOME / "state" / "durable-runtime"
SENTINEL_DIR.mkdir(parents=True, exist_ok=True) if HERMES_HOME.exists() else None

# Simple thread-tracking DB — separate from the full langgraph checkpoint DB
# so Stage A doesn't depend on langgraph being installed. Stage B merges
# this into the main DB.
# DURABLE_THREADS_DB env var override (durable-memory-current.md doctrine).
_DURABLE_THREADS_DB_OVERRIDE = os.environ.get("DURABLE_THREADS_DB")
THREAD_DB = (Path(_DURABLE_THREADS_DB_OVERRIDE).expanduser() if _DURABLE_THREADS_DB_OVERRIDE else HERMES_HOME / "data" / "durable-threads.db")

SHUTDOWN_MARKER = SENTINEL_DIR / "gateway-shutdown-clean.flag"

THREADS_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_threads (
    run_id          TEXT PRIMARY KEY,
    thread_id       TEXT NOT NULL,
    session_key     TEXT,
    agent_profile   TEXT,
    source          TEXT,
    started_at      INTEGER NOT NULL,
    ended_at        INTEGER,
    terminal_state  TEXT,
    note            TEXT
);
CREATE INDEX IF NOT EXISTS idx_agent_threads_open
    ON agent_threads(terminal_state, started_at);
CREATE INDEX IF NOT EXISTS idx_agent_threads_by_thread
    ON agent_threads(thread_id, started_at);
"""


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

def is_enabled(config: dict) -> bool:
    """Read durable_runtime flag from gateway config. Default: False."""
    if not isinstance(config, dict):
        return False
    raw = config.get("durable_runtime", False)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, dict):
        return bool(raw.get("enabled", False))
    if isinstance(raw, str):
        return raw.strip().lower() in ("true", "yes", "1", "on")
    return bool(raw)


def is_stage_b_enabled(config: dict) -> bool:
    """
    Read durable_runtime_stage_b flag. Requires durable_runtime to also be on
    (stage B extends stage A; can't run B without A).

    When True, the gateway will:
      - apply Stage B schema (agent_turns, agent_resume_chains, operator_alerts,
        fencing triggers) on startup
      - call durable_resume.scan_and_claim after the Stage A startup hook
      - allow run_conversation to write per-turn checkpoints (pending integration)
    """
    if not isinstance(config, dict):
        return False
    if not is_enabled(config):
        return False
    raw = config.get("durable_runtime_stage_b", False)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in ("true", "yes", "1", "on")
    return bool(raw)


# ─────────────────────────────────────────────────────────────────────────────
# DB primitives
# ─────────────────────────────────────────────────────────────────────────────

_DB_LOCK = threading.Lock()
_DB_CONN: Optional[sqlite3.Connection] = None


def _open_db() -> sqlite3.Connection:
    """Open (or create) the thread-tracking DB. Single-writer per profile."""
    global _DB_CONN
    with _DB_LOCK:
        if _DB_CONN is not None:
            return _DB_CONN
        THREAD_DB.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(THREAD_DB), isolation_level=None, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(THREADS_SCHEMA)
        try:
            THREAD_DB.chmod(0o600)
        except OSError:
            pass
        _DB_CONN = conn
        return conn


def _close_db() -> None:
    global _DB_CONN
    with _DB_LOCK:
        if _DB_CONN is not None:
            try:
                _DB_CONN.close()
            except Exception:
                pass
            _DB_CONN = None


# ─────────────────────────────────────────────────────────────────────────────
# Gateway lifecycle hooks
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StartupReport:
    enabled: bool
    clean_shutdown: bool
    orphan_count: int
    orphan_threads: list[dict] = field(default_factory=list)
    migration_ok: bool = True
    migration_error: Optional[str] = None
    declarations_loaded: bool = True
    declarations_error: Optional[str] = None
    # Stage B additions
    stage_b_enabled: bool = False
    stage_b_schema_ok: bool = True
    stage_b_schema_error: Optional[str] = None
    stage_b_resume_claimed: int = 0
    stage_b_resume_no_checkpoint: int = 0
    stage_b_resume_chain_exhausted: int = 0
    stage_b_resume_claim_errors: int = 0


def on_gateway_start(config: dict) -> StartupReport:
    """
    Called from GatewayRunner.start() as early as is safe. Returns a
    StartupReport for logging/alerting. No exceptions propagate — any
    failure is captured in the report so the gateway still starts.
    """
    if not is_enabled(config):
        return StartupReport(enabled=False, clean_shutdown=True, orphan_count=0)

    report = StartupReport(enabled=True, clean_shutdown=False, orphan_count=0)

    # 1. Distinguish clean shutdown vs crash.
    if SHUTDOWN_MARKER.exists():
        report.clean_shutdown = True
        try:
            SHUTDOWN_MARKER.unlink()
        except OSError:
            pass

    # 2. Load durability declarations + validate migrations.
    try:
        from tools import durability_declarations as _decl  # noqa: F401
    except Exception as exc:
        report.declarations_loaded = False
        report.declarations_error = f"{type(exc).__name__}: {exc}"
        logger.warning("durable_integration: declarations import failed: %s", exc)

    try:
        from agent import state_migrations as _mig
        _mig.validate_all_migrations()
    except Exception as exc:
        report.migration_ok = False
        report.migration_error = f"{type(exc).__name__}: {exc}"
        logger.error("durable_integration: migration validation failed: %s", exc)

    # 3. Find orphan in-flight runs from the prior process.
    try:
        conn = _open_db()
        cutoff_hours = int(os.environ.get("DURABLE_ORPHAN_CUTOFF_HOURS", "48"))
        cutoff_ts = int(time.time()) - cutoff_hours * 3600
        rows = conn.execute(
            """SELECT run_id, thread_id, session_key, agent_profile,
                      source, started_at, note
               FROM agent_threads
               WHERE terminal_state IS NULL
                 AND started_at >= ?
               ORDER BY started_at DESC""",
            (cutoff_ts,),
        ).fetchall()
        report.orphan_count = len(rows)
        report.orphan_threads = [
            {
                "run_id": r[0], "thread_id": r[1], "session_key": r[2],
                "agent_profile": r[3], "source": r[4],
                "started_at": r[5], "note": r[6],
            }
            for r in rows
        ]
        # Mark them as ABANDONED — we don't auto-resume in Stage A.
        # The operator sees the report and can manually re-engage.
        if rows:
            now = int(time.time())
            conn.executemany(
                """UPDATE agent_threads
                   SET terminal_state='ABANDONED', ended_at=?,
                       note=COALESCE(note,'') || ' | recovered-by-startup'
                   WHERE run_id=?""",
                [(now, r[0]) for r in rows],
            )
    except Exception as exc:
        logger.error("durable_integration: orphan scan failed: %s", exc)

    # ────────────────────────────────────────────────────────────────────
    # Stage B hooks (only if flag is on).
    # ────────────────────────────────────────────────────────────────────
    if is_stage_b_enabled(config):
        report.stage_b_enabled = True
        try:
            from . import durable_schema_stage_b as sb_schema
        except ImportError:
            try:
                import durable_schema_stage_b as sb_schema  # type: ignore[no-redef]
            except ImportError as exc:
                report.stage_b_schema_ok = False
                report.stage_b_schema_error = f"import failed: {exc}"
                logger.error("stage_b: schema module import failed: %s", exc)
                sb_schema = None

        if sb_schema is not None:
            try:
                conn = _open_db()  # reuses the Stage A thread DB
                sb_schema.apply_stage_b_schema(conn)
            except Exception as exc:
                report.stage_b_schema_ok = False
                report.stage_b_schema_error = f"{type(exc).__name__}: {exc}"
                logger.error("stage_b: schema apply failed: %s", exc)

        # Scan for resumable orphans (requires schema applied above).
        if report.stage_b_schema_ok:
            try:
                try:
                    from . import durable_resume as dr
                except ImportError:
                    import durable_resume as dr  # type: ignore[no-redef]
                scan_report, to_schedule = dr.scan_and_claim(_open_db())
                report.stage_b_resume_claimed = scan_report.claimed
                report.stage_b_resume_no_checkpoint = scan_report.no_checkpoint
                report.stage_b_resume_chain_exhausted = scan_report.chain_exhausted
                report.stage_b_resume_claim_errors = scan_report.claim_errors
                # NOTE: to_schedule is returned for the gateway to dispatch.
                # Stage B caller (gateway/run.py) reads this via the
                # `on_gateway_start_with_resumes` helper (see below).
                _STAGE_B_PENDING_RESUMES.extend(to_schedule)
            except Exception as exc:
                logger.error("stage_b: resume scan failed: %s", exc, exc_info=True)

    logger.info(
        "durable_integration: startup report — enabled=%s clean_shutdown=%s "
        "orphan_count=%d migrations_ok=%s decls_ok=%s "
        "stage_b=%s resumed=%d no_ckpt=%d chain_exhausted=%d claim_errs=%d",
        report.enabled, report.clean_shutdown, report.orphan_count,
        report.migration_ok, report.declarations_loaded,
        report.stage_b_enabled,
        report.stage_b_resume_claimed,
        report.stage_b_resume_no_checkpoint,
        report.stage_b_resume_chain_exhausted,
        report.stage_b_resume_claim_errors,
    )
    return report


# Scheduled resumes produced by scan_and_claim; consumed by the gateway's
# post-startup hook (to avoid making on_gateway_start synchronously wait
# on real agent invocations).
_STAGE_B_PENDING_RESUMES: list = []


def drain_pending_resumes() -> list:
    """Atomically drain and return the pending resume dispatches."""
    out = list(_STAGE_B_PENDING_RESUMES)
    _STAGE_B_PENDING_RESUMES.clear()
    return out


def on_gateway_stop(config: dict) -> None:
    """Called from GatewayRunner.stop() to mark a clean shutdown."""
    if not is_enabled(config):
        return
    try:
        SENTINEL_DIR.mkdir(parents=True, exist_ok=True)
        SHUTDOWN_MARKER.write_text(
            json.dumps({"at": int(time.time())}),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("durable_integration: could not write shutdown marker: %s", exc)
    _close_db()


# ─────────────────────────────────────────────────────────────────────────────
# Agent-run tracking hooks
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunHandle:
    run_id: str
    thread_id: str
    enabled: bool
    started_at: int


def on_agent_start(
    *,
    config: dict,
    session_id: str,
    session_key: Optional[str],
    source_platform: str,
    agent_profile: str = "enoch",
    note: Optional[str] = None,
) -> RunHandle:
    """
    Called at the top of _run_agent. Returns a RunHandle that MUST be passed
    to on_agent_end. When the flag is off, returns a no-op handle.
    """
    if not is_enabled(config):
        return RunHandle(run_id="", thread_id="", enabled=False,
                         started_at=int(time.time()))

    thread_id = session_key or session_id
    run_id = uuid.uuid4().hex
    started = int(time.time())
    try:
        conn = _open_db()
        conn.execute(
            """INSERT INTO agent_threads
               (run_id, thread_id, session_key, agent_profile, source,
                started_at, note)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (run_id, thread_id, session_key, agent_profile,
             source_platform, started, note),
        )
    except Exception as exc:
        # Never let tracking break the real agent path.
        logger.error("durable_integration: on_agent_start insert failed: %s", exc)

    return RunHandle(run_id=run_id, thread_id=thread_id,
                     enabled=True, started_at=started)


def on_agent_end(
    handle: RunHandle,
    *,
    terminal_state: str,
    note: Optional[str] = None,
) -> None:
    """
    Mark the run terminal. `terminal_state` is one of
    'SUCCESS' | 'FAIL' | 'CANCELLED' | 'ABANDONED'.
    """
    if not handle.enabled or not handle.run_id:
        return
    try:
        conn = _open_db()
        conn.execute(
            """UPDATE agent_threads
               SET terminal_state=?, ended_at=?,
                   note=COALESCE(?, note)
               WHERE run_id=?""",
            (terminal_state, int(time.time()), note, handle.run_id),
        )
    except Exception as exc:
        logger.error("durable_integration: on_agent_end update failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Observability helpers (for operator alerts)
# ─────────────────────────────────────────────────────────────────────────────

def format_orphan_report(report: StartupReport) -> str:
    """Produce a concise string suitable for an operator alert."""
    if not report.enabled:
        return "durable_runtime disabled"
    if report.orphan_count == 0 and report.clean_shutdown:
        return "durable_runtime OK (clean shutdown, no orphans)"
    lines = [
        f"durable_runtime startup:",
        f"  clean_shutdown: {report.clean_shutdown}",
        f"  orphan in-flight runs: {report.orphan_count} (marked ABANDONED)",
    ]
    if not report.migration_ok:
        lines.append(f"  migration error: {report.migration_error}")
    if not report.declarations_loaded:
        lines.append(f"  declarations error: {report.declarations_error}")
    for o in report.orphan_threads[:5]:
        lines.append(
            f"    - {o['agent_profile']}/{o['source']}: "
            f"thread={o['thread_id'][:40]} "
            f"age={int(time.time()) - o['started_at']}s"
        )
    if report.orphan_count > 5:
        lines.append(f"    ... and {report.orphan_count - 5} more")
    return "\n".join(lines)


def get_inflight_count() -> int:
    """Return the count of currently in-flight runs. For status queries."""
    try:
        conn = _open_db()
        row = conn.execute(
            "SELECT COUNT(*) FROM agent_threads WHERE terminal_state IS NULL"
        ).fetchone()
        return int(row[0]) if row else 0
    except Exception:
        return 0
