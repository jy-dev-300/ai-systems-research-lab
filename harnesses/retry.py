from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FailureDisposition(StrEnum):
    RETRY = "retry"
    FALLBACK = "fallback"
    TERMINATE = "terminate"


class FailureKind(StrEnum):
    MALFORMED_MODEL_OUTPUT = "malformed_model_output"
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_TOOL_ARGUMENTS = "invalid_tool_arguments"
    TOOL_EXCEPTION = "tool_exception"
    MODEL_TIMEOUT = "model_timeout"
    REPEATED_ACTION = "repeated_action"
    LOOP_DETECTED = "loop_detected"


@dataclass(frozen=True)
class RetryDecision:
    disposition: FailureDisposition
    reason: str
    backoff_seconds: float = 0.0


@dataclass(frozen=True)
class RetryContext:
    failure: FailureKind
    attempts: int
    max_retries: int
    repeated_action_count: int = 0


class RetryPolicy:
    """Explicit retry, fallback, and termination policy."""

    def decide(self, context: RetryContext) -> RetryDecision:
        if context.failure is FailureKind.LOOP_DETECTED or context.repeated_action_count >= 3:
            return RetryDecision(
                FailureDisposition.TERMINATE,
                "loop detected; repeating the same action would not make progress",
            )

        if context.attempts >= context.max_retries:
            return RetryDecision(
                FailureDisposition.TERMINATE,
                f"retry budget exhausted after {context.attempts} recovery attempts",
            )

        if context.failure in {
            FailureKind.MALFORMED_MODEL_OUTPUT,
            FailureKind.MODEL_TIMEOUT,
        }:
            return RetryDecision(
                FailureDisposition.RETRY,
                f"{context.failure.value} is transient or may be corrected by regeneration",
                backoff_seconds=min(0.1 * (2**context.attempts), 1.0),
            )

        if context.failure in {
            FailureKind.UNKNOWN_TOOL,
            FailureKind.INVALID_TOOL_ARGUMENTS,
            FailureKind.TOOL_EXCEPTION,
            FailureKind.REPEATED_ACTION,
        }:
            return RetryDecision(
                FailureDisposition.FALLBACK,
                f"feed {context.failure.value} evidence back to the model and "
                "request another action",
            )

        return RetryDecision(FailureDisposition.TERMINATE, "failure has no recovery policy")
