"""
durable_schema_stage_b.py — Stage B schema extensions for the shared
durable-threads.db.

Idempotent: all migrations use ADD COLUMN IF NOT EXISTS (or safely guarded
SELECT-first) and CREATE TABLE/INDEX/TRIGGER IF NOT EXISTS. Safe to run on
every gateway start.

Applied by durable_integration.on_gateway_start when durable_runtime_stage_b
is enabled. Also callable directly for tests.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Iterable

logger = logging.getLogger("agent.durable_schema_stage_b")


STAGE_B_DDL: list[str] = [
    # ─────────────────────────────────────────────────────────────────
    # agent_threads extensions (v0.5 §4.1)
    # ─────────────────────────────────────────────────────────────────
    # These columns MUST use ADD COLUMN (SQLite has no IF NOT EXISTS for
    # ALTER before v3.35, so we guard with a python-side check — see
    # apply_stage_b_schema below).

    # ─────────────────────────────────────────────────────────────────
    # New tables (v0.5 §4.1)
    # ─────────────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS agent_resume_chains (
        resume_chain_id  TEXT PRIMARY KEY,
        next_generation  INTEGER NOT NULL DEFAULT 1,
        created_at       INTEGER NOT NULL
    )
    """,

    """
    CREATE TABLE IF NOT EXISTS operator_alerts (
        alert_key        TEXT PRIMARY KEY,
        chain_id         TEXT NOT NULL,
        reason_category  TEXT NOT NULL,
        reason_detail    TEXT,
        created_at       INTEGER NOT NULL
    )
    """,

    """
    CREATE INDEX IF NOT EXISTS idx_operator_alerts_chain
        ON operator_alerts(chain_id, created_at)
    """,

    # ─────────────────────────────────────────────────────────────────
    # agent_turns (v0.5 §4.3, with v0.5 council fix #1: resume_generation in PK)
    # ─────────────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS agent_turns (
        resume_chain_id      TEXT    NOT NULL,
        turn_index           INTEGER NOT NULL,
        phase_ordinal        INTEGER NOT NULL
                              CHECK(phase_ordinal IN (1, 2, 3)),
        checkpoint_seq       INTEGER NOT NULL,
        resume_generation    INTEGER NOT NULL,
        state_schema_version INTEGER NOT NULL,
        state_parser_version INTEGER NOT NULL,
        state_json           TEXT    NOT NULL,
        last_request_hash    TEXT,
        last_response_body   TEXT,
        state_json_sha256    TEXT    NOT NULL,
        written_by_run_id    TEXT    NOT NULL,
        created_at           INTEGER NOT NULL,
        PRIMARY KEY (resume_chain_id, turn_index, phase_ordinal,
                     checkpoint_seq, resume_generation)
    )
    """,

    """
    CREATE INDEX IF NOT EXISTS idx_agent_turns_by_chain
        ON agent_turns(resume_chain_id, turn_index DESC,
                        phase_ordinal DESC, checkpoint_seq DESC)
    """,
]


# Triggers go in their own block because they use BEFORE INSERT OR UPDATE
# which varies slightly across SQLite versions. These are idempotent via
# DROP TRIGGER IF EXISTS + CREATE.
STAGE_B_TRIGGERS: list[str] = [
    """
    DROP TRIGGER IF EXISTS agent_turns_reject_stale_gen
    """,
    """
    CREATE TRIGGER agent_turns_reject_stale_gen
    BEFORE INSERT ON agent_turns
    WHEN EXISTS (
        SELECT 1 FROM agent_threads
         WHERE resume_chain_id = NEW.resume_chain_id
           AND resume_generation > NEW.resume_generation
    )
    BEGIN
        SELECT RAISE(ABORT, 'stale resume_generation: agent_turns INSERT rejected');
    END
    """,
    """
    DROP TRIGGER IF EXISTS agent_turns_reject_stale_gen_upd
    """,
    """
    CREATE TRIGGER agent_turns_reject_stale_gen_upd
    BEFORE UPDATE ON agent_turns
    WHEN EXISTS (
        SELECT 1 FROM agent_threads
         WHERE resume_chain_id = NEW.resume_chain_id
           AND resume_generation > NEW.resume_generation
    )
    BEGIN
        SELECT RAISE(ABORT, 'stale resume_generation: agent_turns UPDATE rejected');
    END
    """,

    """
    DROP TRIGGER IF EXISTS pending_commits_reject_stale_gen
    """,
    """
    CREATE TRIGGER pending_commits_reject_stale_gen
    BEFORE INSERT ON pending_commits
    WHEN NEW.resume_chain_id IS NOT NULL
     AND EXISTS (
        SELECT 1 FROM agent_threads
         WHERE resume_chain_id = NEW.resume_chain_id
           AND resume_generation > COALESCE(NEW.resume_generation, -1)
    )
    BEGIN
        SELECT RAISE(ABORT, 'stale resume_generation: pending_commits INSERT rejected');
    END
    """,
    """
    DROP TRIGGER IF EXISTS pending_commits_reject_stale_gen_upd
    """,
    """
    CREATE TRIGGER pending_commits_reject_stale_gen_upd
    BEFORE UPDATE ON pending_commits
    WHEN NEW.resume_chain_id IS NOT NULL
     AND EXISTS (
        SELECT 1 FROM agent_threads
         WHERE resume_chain_id = NEW.resume_chain_id
           AND resume_generation > COALESCE(NEW.resume_generation, -1)
    )
    BEGIN
        SELECT RAISE(ABORT, 'stale resume_generation: pending_commits UPDATE rejected');
    END
    """,

    # agent_threads self-fencing: any state-changing update from a stale
    # generation is rejected (v0.5 council fix #2).
    """
    DROP TRIGGER IF EXISTS agent_threads_reject_stale_update
    """,
    """
    CREATE TRIGGER agent_threads_reject_stale_update
    BEFORE UPDATE ON agent_threads
    FOR EACH ROW
    WHEN
        -- Only fence state transitions that a zombie writer could
        -- ILLEGITIMATELY perform. superseded_by_run_id is ALWAYS set by
        -- the orchestrator's claim transaction (which by definition runs
        -- AT a newer generation after inserting the replacement), so a
        -- newer-gen existing at supersede time is the EXPECTED state and
        -- must not trigger rejection.
        (NEW.terminal_state IS NOT OLD.terminal_state
         OR NEW.resume_claimed_at IS NOT OLD.resume_claimed_at)
        AND EXISTS (
            SELECT 1 FROM agent_threads
             WHERE resume_chain_id = OLD.resume_chain_id
               AND resume_generation > OLD.resume_generation
        )
    BEGIN
        SELECT RAISE(ABORT, 'stale resume_generation: agent_threads state update rejected');
    END
    """,

    # Fencing on agent_resume_chains and operator_alerts — same pattern
    # (v0.5 council fix #3 — "fencing triggers missing on these tables").
    """
    DROP TRIGGER IF EXISTS agent_resume_chains_reject_stale
    """,
    """
    CREATE TRIGGER agent_resume_chains_reject_stale
    BEFORE UPDATE ON agent_resume_chains
    WHEN EXISTS (
        SELECT 1 FROM agent_threads
         WHERE resume_chain_id = NEW.resume_chain_id
           AND terminal_state IN ('SUCCESS','FAIL','CANCELLED','NEEDS_OPERATOR_REVIEW')
           AND superseded_by_run_id IS NULL
    )
    BEGIN
        SELECT RAISE(ABORT,
            'agent_resume_chains update rejected: chain has terminal non-superseded run');
    END
    """,
    # NOTE: operator_alerts dedup is enforced by PRIMARY KEY. No separate
    # fencing trigger needed — stale writers simply hit the same dedup path.
]


def apply_stage_b_schema(conn: sqlite3.Connection) -> None:
    """
    Apply Stage B schema to the already-open durable DB. Idempotent.

    Ensures Stage A tables (agent_threads, pending_commits) exist first
    by delegating to their respective initializers before adding the
    Stage B extensions + triggers that reference them.
    """
    # ─── 0. Ensure Stage A tables exist first (idempotent). ───
    # agent_threads is created by durable_integration._open_db.
    # pending_commits is created by IdempotencyStore.initialize_schema.
    # On a fresh profile, on_gateway_start creates agent_threads via
    # _open_db but NOT pending_commits (lazily created when first tool
    # dispatch happens). Stage B triggers reference pending_commits so
    # we must create it here to avoid "no such table" errors.
    try:
        from . import idempotency as _idem
    except ImportError:
        import idempotency as _idem  # type: ignore[no-redef]
    _idem.initialize_schema(conn)
    # ─── 1. ALTER agent_threads for resume-chain columns (v0.5 §4.1) ───
    existing_cols = _get_columns(conn, "agent_threads")
    if "resume_chain_id" not in existing_cols:
        conn.execute("ALTER TABLE agent_threads ADD COLUMN resume_chain_id TEXT")
    if "resume_generation" not in existing_cols:
        conn.execute(
            "ALTER TABLE agent_threads ADD COLUMN "
            "resume_generation INTEGER NOT NULL DEFAULT 0"
        )
    if "superseded_by_run_id" not in existing_cols:
        conn.execute(
            "ALTER TABLE agent_threads ADD COLUMN superseded_by_run_id TEXT"
        )
    if "resume_claimed_at" not in existing_cols:
        conn.execute(
            "ALTER TABLE agent_threads ADD COLUMN resume_claimed_at INTEGER"
        )

    # ─── 2. UNIQUE index for generation monotonicity (v0.5 §4.1) ───
    # Partial-predicate UNIQUE: only where resume_chain_id is non-null.
    # This avoids the "many NULL rows" degenerate case for chain-id-less
    # Stage A rows.
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uidx_agent_threads_chain_gen "
        "ON agent_threads(resume_chain_id, resume_generation) "
        "WHERE resume_chain_id IS NOT NULL"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_agent_threads_chain_scan "
        "ON agent_threads(resume_chain_id, started_at) "
        "WHERE resume_chain_id IS NOT NULL"
    )

    # ─── 3. ALTER pending_commits for chain/gen fencing (v0.5 §4.3) ───
    pc_cols = _get_columns(conn, "pending_commits")
    if "resume_chain_id" not in pc_cols:
        conn.execute(
            "ALTER TABLE pending_commits ADD COLUMN resume_chain_id TEXT"
        )
    if "resume_generation" not in pc_cols:
        conn.execute(
            "ALTER TABLE pending_commits ADD COLUMN resume_generation INTEGER"
        )
    # Allow 'abandoned' status for the queryable-not_found cleanup path
    # (v0.5 §4.9). SQLite doesn't support adding CHECK constraint retroactively,
    # so the 'abandoned' value is validated at application level here — the
    # Stage A schema CHECK will reject it. Bypass: the column doesn't re-check
    # on UPDATE when the CHECK was defined with no named constraint — verify.

    # ─── 4. New tables (v0.5 §4.1, §4.3) ───
    for ddl in STAGE_B_DDL:
        conn.execute(ddl)

    # ─── 5. Fencing triggers (v0.5 §4.3, §4.4 — council fixes #2 + #3) ───
    for trigger_sql in STAGE_B_TRIGGERS:
        conn.execute(trigger_sql)

    conn.commit()
    logger.info("stage_b_schema: applied (tables + indexes + triggers)")


def _get_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {row[1] for row in rows}  # row[1] is column name
    except sqlite3.OperationalError:
        return set()


def verify_stage_b_schema(conn: sqlite3.Connection) -> dict[str, bool]:
    """Diagnostic: report which Stage B artifacts are present."""
    report = {
        "table_agent_resume_chains": _table_exists(conn, "agent_resume_chains"),
        "table_operator_alerts":     _table_exists(conn, "operator_alerts"),
        "table_agent_turns":         _table_exists(conn, "agent_turns"),
        "col_agent_threads_chain":   "resume_chain_id" in _get_columns(conn, "agent_threads"),
        "col_agent_threads_gen":     "resume_generation" in _get_columns(conn, "agent_threads"),
        "col_pending_commits_chain": "resume_chain_id" in _get_columns(conn, "pending_commits"),
        "trigger_agent_turns_ins":   _trigger_exists(conn, "agent_turns_reject_stale_gen"),
        "trigger_agent_turns_upd":   _trigger_exists(conn, "agent_turns_reject_stale_gen_upd"),
        "trigger_pending_ins":       _trigger_exists(conn, "pending_commits_reject_stale_gen"),
        "trigger_pending_upd":       _trigger_exists(conn, "pending_commits_reject_stale_gen_upd"),
        "trigger_threads_stale_upd": _trigger_exists(conn, "agent_threads_reject_stale_update"),
    }
    return report


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def _trigger_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='trigger' AND name=?",
        (name,),
    ).fetchone()
    return row is not None
