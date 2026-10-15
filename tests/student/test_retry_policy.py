import pytest

from harnesses.retry import (
    FailureDisposition,
    FailureKind,
    RetryContext,
    RetryPolicy,
)

pytestmark = pytest.mark.student


def test_retryable_failure_retries_with_budget_remaining() -> None:
    decision = RetryPolicy().decide(
        RetryContext(failure=FailureKind.MODEL_TIMEOUT, attempts=1, max_retries=2)
    )
    assert decision.disposition is FailureDisposition.RETRY


def test_exhausted_retry_budget_terminates() -> None:
    decision = RetryPolicy().decide(
        RetryContext(failure=FailureKind.MODEL_TIMEOUT, attempts=2, max_retries=2)
    )
    assert decision.disposition is FailureDisposition.TERMINATE


def test_detected_loop_does_not_blindly_retry() -> None:
    decision = RetryPolicy().decide(
        RetryContext(
            failure=FailureKind.LOOP_DETECTED,
            attempts=0,
            max_retries=2,
            repeated_action_count=3,
        )
    )
    assert decision.disposition in {FailureDisposition.FALLBACK, FailureDisposition.TERMINATE}
