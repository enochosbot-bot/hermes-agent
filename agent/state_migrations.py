"""
state_migrations.py — versioned state migrations with mandatory fixtures.

Implements spec v0.3 §4.8:

  - Every migration in MIGRATIONS must have a corresponding fixture pair
    (OLD_STATE_SAMPLE, EXPECTED_NEW_STATE) in MIGRATION_FIXTURES.
  - At gateway start, validate_all_migrations() runs each migration against
    its fixture and compares the result to the expected output. A mismatch
    is a STARTUP ERROR — the gateway refuses to start.
  - Migrations apply sequentially; a runtime failure on real state is
    CONTAINED to the single thread (marked needs_operator_review), not
    fatal to the gateway.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Callable

logger = logging.getLogger("agent.state_migrations")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

CURRENT_GRAPH_VERSION = 1  # Phase 1: v1 = baseline. Bump on every state shape change.


# Each migration is a pure function (old_state: dict) -> dict (new_state).
# Key is the TARGET version. E.g. MIGRATIONS[2] takes v1 → v2.
MIGRATIONS: dict[int, Callable[[dict], dict]] = {}


# Fixtures: key is the same TARGET version as MIGRATIONS.
# Value is (sample_old_state, expected_new_state).
# Both sides are dicts; the comparison is structural (see _deep_equal).
MIGRATION_FIXTURES: dict[int, tuple[dict, dict]] = {}


# migration_log table audits applied migrations.
MIGRATION_LOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS migration_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id      TEXT NOT NULL,
    from_version   INTEGER NOT NULL,
    to_version     INTEGER NOT NULL,
    before_hash    TEXT NOT NULL,
    after_hash     TEXT NOT NULL,
    applied_at     INTEGER NOT NULL,
    status         TEXT NOT NULL CHECK(status IN ('success','failed'))
);
CREATE INDEX IF NOT EXISTS idx_migration_log_thread
    ON migration_log(thread_id, applied_at);
"""


def initialize_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(MIGRATION_LOG_SCHEMA)
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Registration helpers (used by tests, and by new migrations as they ship)
# ─────────────────────────────────────────────────────────────────────────────

class MigrationValidationError(Exception):
    """Raised when migration fixtures are missing or produce unexpected output."""


def register_migration(
    *,
    target_version: int,
    migrate: Callable[[dict], dict],
    old_fixture: dict,
    expected_new: dict,
) -> None:
    """
    Register a migration and its mandatory fixture pair.

    This is the canonical way to add a migration — it forces the developer
    to supply the fixture at registration time. No fixture = no registration.
    """
    if target_version <= 1:
        raise MigrationValidationError(
            "target_version must be >= 2 (v1 is the baseline)"
        )
    MIGRATIONS[target_version] = migrate
    MIGRATION_FIXTURES[target_version] = (old_fixture, expected_new)


def clear_registrations_for_tests() -> None:
    MIGRATIONS.clear()
    MIGRATION_FIXTURES.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Fixture validation (runs at gateway start)
# ─────────────────────────────────────────────────────────────────────────────

def _deep_equal(a: Any, b: Any) -> bool:
    """Structural equality ignoring dict key order. No special treatment of lists."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return False
        return all(_deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        return all(_deep_equal(x, y) for x, y in zip(a, b))
    return a == b


def _structural_diff(expected: Any, actual: Any, path: str = "$") -> list[str]:
    """Produce a human-readable diff listing when fixtures mismatch."""
    diffs: list[str] = []
    if type(expected) is not type(actual):
        diffs.append(f"{path}: type mismatch (expected {type(expected).__name__}, got {type(actual).__name__})")
        return diffs
    if isinstance(expected, dict):
        for k in expected:
            if k not in actual:
                diffs.append(f"{path}.{k}: missing in actual")
            else:
                diffs.extend(_structural_diff(expected[k], actual[k], f"{path}.{k}"))
        for k in actual:
            if k not in expected:
                diffs.append(f"{path}.{k}: unexpected key in actual")
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            diffs.append(f"{path}: length mismatch (expected {len(expected)}, got {len(actual)})")
        for i, (x, y) in enumerate(zip(expected, actual)):
            diffs.extend(_structural_diff(x, y, f"{path}[{i}]"))
    else:
        if expected != actual:
            diffs.append(f"{path}: value mismatch (expected {expected!r}, got {actual!r})")
    return diffs


def validate_all_migrations() -> None:
    """
    Run every registered migration against its fixture pair. Raise
    MigrationValidationError on any missing or incorrect fixture.

    Called at gateway start. A failure here prevents the gateway from
    accepting runs — intentional, per spec §4.8.
    """
    # 1. Every migration must have a fixture.
    missing = set(MIGRATIONS.keys()) - set(MIGRATION_FIXTURES.keys())
    if missing:
        raise MigrationValidationError(
            f"migrations without fixtures: {sorted(missing)}. "
            f"Every migration must ship a (old, expected_new) fixture pair."
        )
    # 2. Every fixture must correspond to a migration.
    orphan = set(MIGRATION_FIXTURES.keys()) - set(MIGRATIONS.keys())
    if orphan:
        raise MigrationValidationError(
            f"orphan fixtures without migrations: {sorted(orphan)}"
        )
    # 3. Versions must be contiguous up to current.
    if MIGRATIONS:
        versions = sorted(MIGRATIONS.keys())
        expected = list(range(2, max(versions) + 1))
        if versions != expected:
            raise MigrationValidationError(
                f"migration versions not contiguous: have {versions}, expected {expected}"
            )
    # 4. Each migration produces the expected new state.
    for target_v, migrate in MIGRATIONS.items():
        old_sample, expected_new = MIGRATION_FIXTURES[target_v]
        try:
            actual_new = migrate(dict(old_sample))  # pass a copy
        except Exception as exc:
            raise MigrationValidationError(
                f"migration v{target_v} raised on fixture: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not _deep_equal(expected_new, actual_new):
            diffs = _structural_diff(expected_new, actual_new)
            diff_text = "\n  ".join(diffs[:20])
            if len(diffs) > 20:
                diff_text += f"\n  ... and {len(diffs) - 20} more"
            raise MigrationValidationError(
                f"migration v{target_v} produced unexpected output from fixture:\n"
                f"  {diff_text}"
            )
    logger.info(
        "state_migrations: validated %d migrations against fixtures",
        len(MIGRATIONS),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sequential migration application
# ─────────────────────────────────────────────────────────────────────────────

def migrate_state_to_current(
    state: dict,
    *,
    from_version: int,
    to_version: int = CURRENT_GRAPH_VERSION,
) -> dict:
    """
    Apply sequential migrations from `from_version` → `to_version`.

    Raises MigrationValidationError if a required migration is missing
    (which should have been caught at startup, but defensive).
    Re-raises any exception from a migration — caller decides whether to
    contain to a single thread (mark needs_operator_review) or propagate.
    """
    if from_version > to_version:
        raise MigrationValidationError(
            f"refusing forward-to-backward migration: from v{from_version} > to v{to_version}. "
            "Gateway was rolled back from a newer deploy; redeploy newer code or force-archive."
        )
    current = dict(state)
    for v in range(from_version + 1, to_version + 1):
        if v not in MIGRATIONS:
            raise MigrationValidationError(
                f"missing migration for v{v} during resume; this should have been caught at startup"
            )
        current = MIGRATIONS[v](current)
    return current


def log_migration(
    conn: sqlite3.Connection,
    *,
    thread_id: str,
    from_version: int,
    to_version: int,
    before_hash: str,
    after_hash: str,
    status: str,
    applied_at: int,
) -> None:
    """Record an applied migration for audit."""
    conn.execute(
        """INSERT INTO migration_log
           (thread_id, from_version, to_version, before_hash, after_hash,
            applied_at, status)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (thread_id, from_version, to_version, before_hash, after_hash,
         applied_at, status),
    )
