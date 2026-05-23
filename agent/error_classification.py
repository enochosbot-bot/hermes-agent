"""
error_classification.py — classify failures for retry decisions.

Per spec v0.3 §4.2.1: every thrown exception, failed HTTP call, or tool error
is classified as 'transient' | 'permanent' | 'unknown'. Retry budgets apply
only within 'transient'. 'unknown' is fail-safe — treated as 'permanent' to
avoid retry loops on unrecognized conditions.

The classification is a pure function of (exception_class, status_code, tool_name).
No side effects, no state. Profiles can supply an override table via register_overrides().
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Optional, Type

logger = logging.getLogger("agent.error_classification")

ErrorClass = Literal["transient", "permanent", "unknown"]


@dataclass(frozen=True)
class ErrorSignal:
    """A normalized error description passed to the classifier."""
    exception_class: Optional[str] = None  # e.g. "TimeoutError", "httpx.ReadError"
    status_code: Optional[int] = None      # HTTP status if applicable
    tool_name: Optional[str] = None        # tool that raised, if any
    message: Optional[str] = None          # free-form, not used for decision but logged


# Default HTTP status classification.
# Spec: 4xx = permanent EXCEPT 408 (request timeout) and 429 (rate limit) = transient.
# 5xx = transient (service-side, retry).
def _classify_status_code(status: Optional[int]) -> Optional[ErrorClass]:
    if status is None:
        return None
    if status in (408, 429):
        return "transient"
    if 400 <= status < 500:
        return "permanent"
    if 500 <= status < 600:
        return "transient"
    return None  # 2xx/3xx shouldn't reach classifier


# Default exception-class classification.
# Names are string-matched (not isinstance) so this works across SDK boundaries
# without forcing imports on callers.
DEFAULT_EXCEPTION_TABLE: dict[str, ErrorClass] = {
    # transient — network, rate, server
    "TimeoutError":              "transient",
    "TimeoutException":          "transient",
    "ConnectionError":           "transient",
    "ConnectionResetError":      "transient",
    "ConnectionRefusedError":    "transient",
    "ReadError":                 "transient",
    "ReadTimeout":               "transient",
    "RemoteProtocolError":       "transient",
    "ServerError":               "transient",
    "APIConnectionError":        "transient",  # openai sdk
    "RateLimitError":            "transient",
    "OperationalError":          "transient",  # sqlite transient lock

    # permanent — auth, validation, schema, budget
    "AuthenticationError":       "permanent",
    "PermissionDeniedError":     "permanent",
    "InvalidRequestError":       "permanent",
    "BadRequestError":           "permanent",
    "ValidationError":           "permanent",
    "NotFoundError":             "permanent",
    "SchemaValidationError":     "permanent",
    "JSONDecodeError":           "permanent",  # model returned malformed json — let handler retry via error-feedback, not blind retry
    "TypeError":                 "permanent",
    "ValueError":                "permanent",
    "KeyError":                  "permanent",
    "AttributeError":            "permanent",
    "BudgetExhaustedError":      "permanent",
    "IdempotencyCollisionError": "permanent",

    # keep-the-process-alive permanent
    "KeyboardInterrupt":         "permanent",
    "SystemExit":                "permanent",
}


# Per-tool overrides. Key: tool_name. Value: table like DEFAULT_EXCEPTION_TABLE.
# Populated by register_tool_overrides() at build time.
_TOOL_OVERRIDES: dict[str, dict[str, ErrorClass]] = {}


def register_tool_overrides(tool_name: str, table: dict[str, ErrorClass]) -> None:
    """Install per-tool exception classifications that override the default."""
    _TOOL_OVERRIDES[tool_name] = dict(table)


def classify(signal: ErrorSignal) -> ErrorClass:
    """
    Return the error class for a given signal.

    Precedence (first match wins):
      1. Tool-specific override (if tool_name present and override registered).
      2. HTTP status code (if present).
      3. Exception class name (default table).
      4. 'unknown' → treated as 'permanent' by the graph, but logged as 'unknown'.
    """
    # 1. Tool override
    if signal.tool_name and signal.tool_name in _TOOL_OVERRIDES:
        tbl = _TOOL_OVERRIDES[signal.tool_name]
        if signal.exception_class and signal.exception_class in tbl:
            return tbl[signal.exception_class]

    # 2. HTTP status
    status_verdict = _classify_status_code(signal.status_code)
    if status_verdict is not None:
        return status_verdict

    # 3. Exception class
    if signal.exception_class and signal.exception_class in DEFAULT_EXCEPTION_TABLE:
        return DEFAULT_EXCEPTION_TABLE[signal.exception_class]

    # 4. Unknown — log loudly.
    logger.warning(
        "error_classification: unknown signal — treating as permanent. "
        "exception_class=%r status=%r tool=%r message=%r",
        signal.exception_class, signal.status_code, signal.tool_name, signal.message,
    )
    return "unknown"


def classify_exception(exc: BaseException, *, tool_name: Optional[str] = None,
                       status_code: Optional[int] = None) -> ErrorClass:
    """Convenience wrapper: pass a live exception."""
    return classify(ErrorSignal(
        exception_class=type(exc).__name__,
        status_code=status_code,
        tool_name=tool_name,
        message=str(exc)[:500],
    ))


def is_retryable(error_class: ErrorClass) -> bool:
    """
    Graph-facing predicate: should this error be retried?

    Only 'transient' is retryable. 'permanent' and 'unknown' both short-circuit
    to terminal FAIL — 'unknown' is treated as permanent (fail-safe) to avoid
    infinite loops on errors we don't recognize.
    """
    return error_class == "transient"
