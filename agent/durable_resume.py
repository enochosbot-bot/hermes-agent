"""
durable_resume.py — scan_and_claim, claim_and_create_replacement,
load_latest_valid_checkpoint, operator-alert dedup, sentinel-file escalation.

Implements spec v0.5 §4.2, §4.10, §4.11, §4.12 with the council fixes from
v0.5 review: (1) checkpoint validated inside claim, (2) CAS allocator via
agent_resume_chains, (3) IntegrityError retries instead of surrendering,
(4) _mark_needs_operator_review atomic with terminal_state='ABANDONED'
guard, (5) operator_alerts table with PK dedup.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

try:
    from . import durable_state as ds
except ImportError:
    import durable_state as ds  # type: ignore[no-redef]

logger = logging.getLogger("agent.durable_resume")


# ─────────────────────────────────────────────────────────────────────────────
# Constants (spec §10 + council answers)
# ─────────────────────────────────────────────────────────────────────────────

MAX_CHAIN_LENGTH = 3                 # v0.5 §4.10
AUTO_RESUME_WINDOW_S = 3600          # 1 hour (council-answered default)
PROBE_TIMEOUT_S = 10                 # queryable reconciler probe timeout
CLAIM_MAX_ATTEMPTS = 3               # v0.5 §4.2
CLAIM_BACKOFF_BASE_S = 0.05

# Single-process mutex over scan_and_claim — prevents intra-process threads
# from racing. DB-level fencing protects cross-process scenarios.
_RESUME_MUTEX = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Data shapes
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class OrphanRow:
    run_id: str
    thread_id: str
    session_key: Optional[str]
    agent_profile: str
    source: str
    resume_chain_id: str
    resume_generation: int
    started_at: int
    delivery_epoch: Optional[int] = None


@dataclasses.dataclass
class RunHandle:
    run_id: str
    thread_id: str
    resume_chain_id: str
    resume_generation: int
    started_at: int


@dataclasses.dataclass
class ResumeScanReport:
    orphans_scanned: int = 0
    claimed: int = 0
    no_checkpoint: int = 0
    chain_exhausted: int = 0
    stale_rollback: int = 0
    claim_errors: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Operator alerts with dedup (v0.5 §4.10)
# ─────────────────────────────────────────────────────────────────────────────

def emit_operator_alert_with_dedup(
    conn: sqlite3.Connection,
    *,
    chain_id: str,
    reason_category: str,
    reason_detail: str = "",
    deliver_fn: Optional[Callable[[str, str, str], None]] = None,
) -> bool:
    """
    Insert into operator_alerts with PK conflict = already-alerted (skip).
    Returns True if a NEW alert was emitted, False if deduped.

    `deliver_fn(chain_id, reason_category, reason_detail)` is called only on
    a new alert. Callers pass their platform-specific sender here.
    """
    alert_key = hashlib.sha256(
        f"{chain_id}:{reason_category}".encode("utf-8")
    ).hexdigest()
    try:
        conn.execute(
            "INSERT INTO operator_alerts "
            "(alert_key, chain_id, reason_category, reason_detail, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (alert_key, chain_id, reason_category, reason_detail, int(time.time())),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        # Duplicate — already alerted for this (chain, reason).
        return False
    except sqlite3.OperationalError as exc:
        # DB unavailable — fall through to sentinel file.
        logger.error("operator_alerts insert failed: %s; writing sentinel", exc)
        _write_escalation_sentinel(chain_id, reason_category, reason_detail)
        return True   # treat as emitted (operator will see sentinel)

    if deliver_fn is not None:
        try:
            deliver_fn(chain_id, reason_category, reason_detail)
        except Exception as exc:
            logger.error("operator alert delivery failed for %s/%s: %s",
                         chain_id, reason_category, exc)
    return True


def _write_escalation_sentinel(chain_id: str, reason_category: str,
                                reason_detail: str) -> None:
    """Fallback when DB is unavailable (spec v0.5 §4.12)."""
    home = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
    sentinel_dir = home / "state" / "durable-escalation-failures"
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    path = sentinel_dir / f"{chain_id}.json"
    try:
        path.write_text(json.dumps({
            "chain_id": chain_id,
            "reason_category": reason_category,
            "reason_detail": reason_detail,
            "created_at": int(time.time()),
        }, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.critical("escalation sentinel write FAILED: %s "
                        "(chain=%s reason=%s)", exc, chain_id, reason_category)


# ─────────────────────────────────────────────────────────────────────────────
# Atomic claim+insert+supersede with CAS allocator (v0.5 §4.2)
# ─────────────────────────────────────────────────────────────────────────────

def claim_and_create_replacement(
    conn: sqlite3.Connection,
    orphan: OrphanRow,
    *,
    now_fn: Callable[[], int] = lambda: int(time.time()),
    new_run_id_fn: Callable[[], str] = lambda: uuid.uuid4().hex,
) -> Optional[RunHandle]:
    """
    Atomically:
      1. Claim the orphan (UPDATE ... WHERE resume_claimed_at IS NULL).
      2. CAS-allocate the next generation via agent_resume_chains.
      3. INSERT replacement run_row.
      4. UPDATE orphan.superseded_by_run_id.

    On IntegrityError (lost race despite CAS): retry up to CLAIM_MAX_ATTEMPTS.
    On OperationalError (lock contention): retry with backoff.
    On 0-rowcount claim: return None (another claimant won, or state isn't
    ABANDONED anymore).
    """
    for attempt in range(CLAIM_MAX_ATTEMPTS):
        # Explicit transaction — connection is in autocommit mode
        # (isolation_level=None) to allow other gateways to read via WAL
        # between our writes. We manage BEGIN/COMMIT/ROLLBACK manually.
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            logger.info("claim attempt %d: BEGIN IMMEDIATE busy: %s", attempt + 1, exc)
            time.sleep(CLAIM_BACKOFF_BASE_S * (2 ** attempt))
            continue

        try:
            # Step 1: claim.
            cur = conn.execute(
                "UPDATE agent_threads "
                "SET resume_claimed_at = ? "
                "WHERE run_id = ? "
                "  AND resume_claimed_at IS NULL "
                "  AND terminal_state = 'ABANDONED'",
                (now_fn(), orphan.run_id),
            )
            if cur.rowcount == 0:
                conn.execute("ROLLBACK")
                return None

            # Step 2: CAS-allocate.
            allocated_gen = _cas_next_generation(conn, orphan.resume_chain_id)

            # Step 3: insert replacement.
            new_run_id = new_run_id_fn()
            conn.execute(
                "INSERT INTO agent_threads "
                "(run_id, thread_id, session_key, agent_profile, source, "
                " started_at, resume_chain_id, resume_generation, note) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    new_run_id,
                    orphan.thread_id,
                    orphan.session_key,
                    orphan.agent_profile,
                    orphan.source,
                    now_fn(),
                    orphan.resume_chain_id,
                    allocated_gen,
                    f"resume of {orphan.run_id}",
                ),
            )

            # Step 4: supersede orphan.
            conn.execute(
                "UPDATE agent_threads "
                "SET superseded_by_run_id = ? "
                "WHERE run_id = ?",
                (new_run_id, orphan.run_id),
            )

            conn.execute("COMMIT")

            return RunHandle(
                run_id=new_run_id,
                thread_id=orphan.thread_id,
                resume_chain_id=orphan.resume_chain_id,
                resume_generation=allocated_gen,
                started_at=now_fn(),
            )

        except sqlite3.IntegrityError as exc:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.OperationalError:
                pass
            logger.info(
                "claim attempt %d IntegrityError: %s; retry",
                attempt + 1, exc,
            )
            time.sleep(CLAIM_BACKOFF_BASE_S * (2 ** attempt))
            continue

        except sqlite3.OperationalError as exc:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.OperationalError:
                pass
            logger.info(
                "claim attempt %d OperationalError: %s; retry",
                attempt + 1, exc,
            )
            time.sleep(CLAIM_BACKOFF_BASE_S * (2 ** attempt))
            continue

        except Exception:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.OperationalError:
                pass
            raise

    logger.error("claim_and_create_replacement exhausted %d attempts for %s",
                 CLAIM_MAX_ATTEMPTS, orphan.run_id)
    emit_operator_alert_with_dedup(
        conn, chain_id=orphan.resume_chain_id,
        reason_category="claim_exhausted",
        reason_detail=f"claim failed {CLAIM_MAX_ATTEMPTS}x for run={orphan.run_id}",
    )
    return None


def _cas_next_generation(conn: sqlite3.Connection, chain_id: str) -> int:
    """
    Atomically increment agent_resume_chains.next_generation and return
    the ALLOCATED generation (i.e. the value before increment).

    First call for a chain creates the row with next_generation=1 and
    allocates generation 0 (reserved for the initial run — which should
    already exist; if not, we still allocate 0 and let the INSERT collide
    harmlessly with whatever already holds that slot).

    Uses ON CONFLICT DO UPDATE + RETURNING for SQLite 3.35+. Falls back to
    two-step for older SQLite.
    """
    try:
        # Preferred path: UPSERT + RETURNING (SQLite 3.35+).
        row = conn.execute(
            "INSERT INTO agent_resume_chains (resume_chain_id, next_generation, created_at) "
            "VALUES (?, 2, ?) "
            "ON CONFLICT(resume_chain_id) DO UPDATE SET "
            "  next_generation = next_generation + 1 "
            "RETURNING next_generation - 1",
            (chain_id, int(time.time())),
        ).fetchone()
        return int(row[0])
    except sqlite3.OperationalError:
        # Older SQLite — fall back to two-step under BEGIN IMMEDIATE lock.
        cur = conn.execute(
            "SELECT next_generation FROM agent_resume_chains WHERE resume_chain_id=?",
            (chain_id,),
        ).fetchone()
        if cur is None:
            conn.execute(
                "INSERT INTO agent_resume_chains "
                "(resume_chain_id, next_generation, created_at) VALUES (?, 2, ?)",
                (chain_id, int(time.time())),
            )
            return 1
        allocated = int(cur[0])
        conn.execute(
            "UPDATE agent_resume_chains SET next_generation = next_generation + 1 "
            "WHERE resume_chain_id=?",
            (chain_id,),
        )
        return allocated


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint load with corrupt-row fallback (v0.5 §4.11)
# ─────────────────────────────────────────────────────────────────────────────

def load_latest_valid_checkpoint(
    conn: sqlite3.Connection,
    chain_id: str,
) -> Optional[dict]:
    """
    Iterate all rows for the chain newest-first (turn, phase, seq, gen),
    return the first whose state_json sha256 matches AND state_parser_version
    <= CURRENT. If state_parser_version > CURRENT (rollback case), return
    None immediately (per v0.5 §4.11).

    Returns the loaded state dict (after migrations) or None.
    """
    cur = conn.execute(
        """
        SELECT state_json, state_json_sha256, state_parser_version,
               state_schema_version, turn_index, phase_ordinal,
               checkpoint_seq, resume_generation
          FROM agent_turns
         WHERE resume_chain_id = ?
         ORDER BY turn_index      DESC,
                  phase_ordinal   DESC,
                  checkpoint_seq  DESC,
                  resume_generation DESC
        """,
        (chain_id,),
    )

    for row in cur:
        (state_json, expected_sha, parser_ver, schema_ver,
         turn_index, phase_ord, chk_seq, gen) = row

        # Rollback check — refuse immediately, don't fall back to older row.
        if parser_ver > ds.CURRENT_STATE_PARSER_VERSION:
            logger.error(
                "load_latest_valid_checkpoint: refusing resume for chain %s: "
                "state_parser_version %d > current %d (rollback?)",
                chain_id, parser_ver, ds.CURRENT_STATE_PARSER_VERSION,
            )
            return None

        try:
            state = ds.deserialize(state_json, expected_sha256=expected_sha)
        except ds.CorruptStateError as exc:
            logger.warning(
                "load_latest_valid_checkpoint: skipping corrupt row "
                "chain=%s turn=%d phase=%d seq=%d gen=%d: %s",
                chain_id, turn_index, phase_ord, chk_seq, gen, exc,
            )
            continue

        try:
            state = ds.migrate_for_resume(state)
            ds.verify_resumable(state)
        except (ds.MigrationMissing, ds.ResumeStateIncompleteError,
                ds.StateParserRollbackError) as exc:
            logger.warning(
                "load_latest_valid_checkpoint: skipping row "
                "chain=%s turn=%d phase=%d: %s",
                chain_id, turn_index, phase_ord, exc,
            )
            continue

        return state

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Resume scan orchestration (v0.5 §4.10)
# ─────────────────────────────────────────────────────────────────────────────

def scan_and_claim(
    conn: sqlite3.Connection,
    *,
    deliver_alert: Optional[Callable[[str, str, str], None]] = None,
    now_fn: Callable[[], int] = lambda: int(time.time()),
    auto_resume_window_s: int = AUTO_RESUME_WINDOW_S,
) -> tuple[ResumeScanReport, list[tuple[RunHandle, dict]]]:
    """
    Find orphan runs eligible for resume, validate their checkpoints,
    atomically claim and create replacement runs.

    Returns (report, [(handle, state), ...]) — caller dispatches each
    replacement via the gateway's run invocation path.

    Validation happens BEFORE claim (v0.5 council fix #5): chains with no
    valid checkpoint are marked NEEDS_OPERATOR_REVIEW without polluting
    the chain with a doomed replacement.
    """
    report = ResumeScanReport()
    to_schedule: list[tuple[RunHandle, dict]] = []

    with _RESUME_MUTEX:
        cutoff = now_fn() - auto_resume_window_s
        orphans_raw = conn.execute(
            """
            SELECT run_id, thread_id, session_key, agent_profile, source,
                   resume_chain_id, resume_generation, started_at
              FROM agent_threads
             WHERE terminal_state = 'ABANDONED'
               AND superseded_by_run_id IS NULL
               AND started_at > ?
               AND resume_chain_id IS NOT NULL
             ORDER BY started_at ASC
            """,
            (cutoff,),
        ).fetchall()
        report.orphans_scanned = len(orphans_raw)

        for r in orphans_raw:
            orphan = OrphanRow(
                run_id=r[0], thread_id=r[1], session_key=r[2],
                agent_profile=r[3], source=r[4], resume_chain_id=r[5],
                resume_generation=int(r[6]), started_at=int(r[7]),
            )

            # Chain-length bound check.
            chain_len = conn.execute(
                "SELECT COUNT(*) FROM agent_threads WHERE resume_chain_id=?",
                (orphan.resume_chain_id,),
            ).fetchone()[0]
            if chain_len >= MAX_CHAIN_LENGTH:
                _mark_needs_operator_review_atomic(conn, orphan)
                emit_operator_alert_with_dedup(
                    conn,
                    chain_id=orphan.resume_chain_id,
                    reason_category="chain_exhausted",
                    reason_detail=f"chain at {chain_len} runs, max {MAX_CHAIN_LENGTH}",
                    deliver_fn=deliver_alert,
                )
                report.chain_exhausted += 1
                continue

            # Validate checkpoint BEFORE claim.
            state = load_latest_valid_checkpoint(conn, orphan.resume_chain_id)
            if state is None:
                _mark_needs_operator_review_atomic(conn, orphan)
                emit_operator_alert_with_dedup(
                    conn,
                    chain_id=orphan.resume_chain_id,
                    reason_category="no_valid_checkpoint",
                    reason_detail=f"no resumable state for run {orphan.run_id}",
                    deliver_fn=deliver_alert,
                )
                report.no_checkpoint += 1
                continue

            handle = claim_and_create_replacement(conn, orphan, now_fn=now_fn)
            if handle is None:
                report.claim_errors += 1
                continue

            to_schedule.append((handle, state))
            report.claimed += 1

    return report, to_schedule


def _mark_needs_operator_review_atomic(
    conn: sqlite3.Connection,
    orphan: OrphanRow,
) -> None:
    """
    Transition ABANDONED → NEEDS_OPERATOR_REVIEW, guarded by current state.
    v0.5 council fix #5: WHERE terminal_state='ABANDONED' ensures we don't
    race with another transition.
    """
    try:
        cur = conn.execute(
            "UPDATE agent_threads "
            "SET terminal_state = 'NEEDS_OPERATOR_REVIEW', ended_at = ? "
            "WHERE run_id = ? "
            "  AND terminal_state = 'ABANDONED' "
            "  AND resume_generation = ?",
            (int(time.time()), orphan.run_id, orphan.resume_generation),
        )
        conn.commit()
        if cur.rowcount == 0:
            logger.warning(
                "_mark_needs_operator_review_atomic: no-op for %s "
                "(state already advanced or generation changed)",
                orphan.run_id,
            )
    except sqlite3.OperationalError as exc:
        logger.error(
            "_mark_needs_operator_review_atomic failed for %s: %s; "
            "falling back to sentinel",
            orphan.run_id, exc,
        )
        _write_escalation_sentinel(
            orphan.resume_chain_id,
            "needs_operator_review_write_failed",
            f"run={orphan.run_id} err={exc}",
        )
