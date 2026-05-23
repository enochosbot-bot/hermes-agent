"""
durable_state.py — Stage B state shape, serialization, and versioning.

Implements spec v0.5 §4.5 + §4.4:

  - StageBState TypedDict with full branch-determinant coverage.
  - Two independent version numbers:
      * state_schema_version  — bumped on shape change (add/remove fields).
      * state_parser_version  — bumped on deserialization-logic change.
  - sha256 of the serialized state for partial-write detection.
  - Rollback-incompatibility rule: refuse to resume a state whose
    state_parser_version > CURRENT_STATE_PARSER_VERSION.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Callable, Optional, TypedDict

logger = logging.getLogger("agent.durable_state")


# ─────────────────────────────────────────────────────────────────────────────
# Versions
# ─────────────────────────────────────────────────────────────────────────────

# Bump ONLY when the dict shape of StageBState changes (field added/removed/
# semantics shift within a field). Old schema versions must have a registered
# migration in SCHEMA_MIGRATIONS.
CURRENT_STATE_SCHEMA_VERSION = 1

# Bump ONLY when the deserialization-logic / field interpretation changes
# in a way that would produce incorrect results if read by older code.
# This is separate from schema because shape-preserving semantic shifts
# still require a parser bump.
CURRENT_STATE_PARSER_VERSION = 1


# ─────────────────────────────────────────────────────────────────────────────
# State shape
# ─────────────────────────────────────────────────────────────────────────────

class StageBState(TypedDict, total=False):
    """
    Per-turn checkpoint state for run_conversation resumption.

    Every field listed in BRANCH_DETERMINANTS below MUST be present at
    resume time or `resume_state` loading fails hard (§4.5 invariant).
    """
    # Schema metadata
    state_schema_version: int
    state_parser_version: int

    # Identity + lineage + fencing
    resume_chain_id: str
    resume_generation: int
    run_id: str              # the run that WROTE this state
    thread_id: str
    session_id: str
    agent_profile: str
    delivery_epoch: int      # FROZEN for the chain's lifetime

    # Conversation
    messages: list[dict]
    system_message: Optional[str]
    ephemeral_system_prompt_hash: Optional[str]
    model: str

    # Turn tracking
    turn_index: int
    max_turns: int
    api_call_count: int

    # Branch determinants — absence at resume = hard error
    provider_name: str
    provider_sdk_version: str
    api_mode: str            # "codex_responses" | "anthropic_messages" | "chat"
    streaming_enabled: bool
    tool_dispatch_mode: str  # "sequential" | "concurrent" (Stage B: sequential only)
    retry_state: dict
    approval_state: Optional[dict]
    compression_state: Optional[dict]
    code_version: str        # audit only — does NOT gate resume

    # Tool state (for in-flight turn)
    pending_tool_calls: list[dict]    # with logical_action_id attached
    completed_tool_calls: list[dict]  # results from current turn
    last_error_class: Optional[str]

    # WAL capture (only at post_response checkpoints)
    last_request_hash: Optional[str]
    last_response_body: Optional[dict]

    # Metadata
    started_at: int
    last_checkpoint_at: int


# Fields that affect control flow on resume. Missing any one = hard error.
BRANCH_DETERMINANTS: tuple[str, ...] = (
    "resume_chain_id", "resume_generation", "run_id", "thread_id",
    "agent_profile", "delivery_epoch",
    "messages", "model",
    "turn_index", "api_call_count",
    "provider_name", "api_mode", "streaming_enabled", "tool_dispatch_mode",
)


class ResumeStateIncompleteError(Exception):
    """Raised when resume state is missing branch-determinant fields."""


class StateParserRollbackError(Exception):
    """Raised when stored state has state_parser_version > CURRENT (rollback)."""


class CorruptStateError(Exception):
    """Raised when state_json fails sha256 verification."""


# ─────────────────────────────────────────────────────────────────────────────
# Serialization
# ─────────────────────────────────────────────────────────────────────────────

def serialize(state: StageBState) -> tuple[str, str]:
    """
    Serialize state to (json_text, sha256_hex). The caller stores both in
    agent_turns; on load, sha256 must match or the row is treated as corrupt.

    Uses deterministic serialization (sorted keys, no whitespace) so sha256
    is stable across identical states.
    """
    # Always include the current schema + parser versions in the serialized
    # form, even if the caller forgot. This guarantees we can always version-
    # check on read.
    state_with_versions = dict(state)
    state_with_versions.setdefault("state_schema_version",
                                    CURRENT_STATE_SCHEMA_VERSION)
    state_with_versions.setdefault("state_parser_version",
                                    CURRENT_STATE_PARSER_VERSION)
    json_text = json.dumps(state_with_versions, sort_keys=True,
                           separators=(",", ":"), default=_json_default)
    digest = hashlib.sha256(json_text.encode("utf-8")).hexdigest()
    return json_text, digest


def _json_default(obj: Any) -> Any:
    """Fallback for non-JSON-native objects (datetime, Path, etc.)."""
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


def deserialize(json_text: str, expected_sha256: Optional[str] = None) -> dict:
    """
    Deserialize state. If expected_sha256 is provided, verify; raise
    CorruptStateError on mismatch.

    Does NOT validate branch-determinants or parser version — those are
    separate steps (see verify_resumable).
    """
    if expected_sha256 is not None:
        actual = hashlib.sha256(json_text.encode("utf-8")).hexdigest()
        if actual != expected_sha256:
            raise CorruptStateError(
                f"state_json sha256 mismatch: expected {expected_sha256[:16]}..., "
                f"got {actual[:16]}..."
            )
    try:
        return json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise CorruptStateError(f"state_json invalid: {exc}") from exc


def verify_resumable(state: dict) -> None:
    """
    Check that state is resumable under the current code version. Raises:
      - StateParserRollbackError  if state_parser_version > CURRENT
      - ResumeStateIncompleteError if any BRANCH_DETERMINANTS field is missing

    Call AFTER any needed schema/parser migrations have been applied.
    """
    stored_parser = state.get("state_parser_version",
                              CURRENT_STATE_PARSER_VERSION)
    if stored_parser > CURRENT_STATE_PARSER_VERSION:
        raise StateParserRollbackError(
            f"state_parser_version {stored_parser} > current "
            f"{CURRENT_STATE_PARSER_VERSION}: this looks like a service rollback. "
            "Redeploy newer code or force-archive this chain manually."
        )

    missing = [f for f in BRANCH_DETERMINANTS if f not in state]
    if missing:
        raise ResumeStateIncompleteError(
            f"state missing branch-determinants: {missing}. Resume cannot proceed."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Migrations
# ─────────────────────────────────────────────────────────────────────────────

# Keyed by the TARGET version; value is a function that takes the v(N-1)
# state dict and returns the v(N) state dict. Fails loud if any migration
# is missing when we try to traverse through it.

SchemaMigration = Callable[[dict], dict]
ParserMigration = Callable[[dict], dict]

SCHEMA_MIGRATIONS: dict[int, SchemaMigration] = {}
PARSER_MIGRATIONS: dict[int, ParserMigration] = {}


class MigrationMissing(Exception):
    """Raised when apply_migrations can't traverse from stored to current."""


def apply_schema_migrations(state: dict) -> dict:
    """
    Apply schema migrations in order from stored version to current.
    Idempotent: if already at current, returns state unchanged.
    """
    current = state.copy()
    stored = current.get("state_schema_version", 1)
    while stored < CURRENT_STATE_SCHEMA_VERSION:
        next_v = stored + 1
        if next_v not in SCHEMA_MIGRATIONS:
            raise MigrationMissing(
                f"no schema migration for v{stored} -> v{next_v}"
            )
        current = SCHEMA_MIGRATIONS[next_v](current)
        current["state_schema_version"] = next_v
        stored = next_v
    return current


def apply_parser_migrations(state: dict) -> dict:
    """Apply parser migrations in order from stored version to current."""
    current = state.copy()
    stored = current.get("state_parser_version", 1)
    while stored < CURRENT_STATE_PARSER_VERSION:
        next_v = stored + 1
        if next_v not in PARSER_MIGRATIONS:
            raise MigrationMissing(
                f"no parser migration for v{stored} -> v{next_v}"
            )
        current = PARSER_MIGRATIONS[next_v](current)
        current["state_parser_version"] = next_v
        stored = next_v
    return current


def migrate_for_resume(state: dict) -> dict:
    """Apply schema + parser migrations to bring state to current code."""
    return apply_parser_migrations(apply_schema_migrations(state))


# ─────────────────────────────────────────────────────────────────────────────
# Phase helpers
# ─────────────────────────────────────────────────────────────────────────────

PHASE_PRE_CALL      = 1
PHASE_POST_RESPONSE = 2
PHASE_POST_TOOLS    = 3


def phase_name(ordinal: int) -> str:
    return {1: "pre_call", 2: "post_response", 3: "post_tools"}.get(
        ordinal, f"unknown({ordinal})"
    )
