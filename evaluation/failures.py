from dataclasses import dataclass
from enum import StrEnum

from benchmarks.models import BenchmarkTask
from evaluation.metrics import evaluate_run
from harnesses.contracts import HarnessResult, RunStatus
from tracing.events import TraceEventType


class FailureCategory(StrEnum):
    WRONG_TOOL = "wrong_tool"
    INVALID_ARGUMENTS = "invalid_arguments"
    TOOL_FAILURE = "tool_failure"
    CONTEXT_FAILURE = "context_failure"
    MALFORMED_OUTPUT = "malformed_output"
    REPEATED_ACTION = "repeated_action"
    LOOP = "loop"
    RETRY_EXHAUSTION = "retry_exhaustion"
    PREMATURE_TERMINATION = "premature_termination"
    INCORRECT_FINAL_ANSWER = "incorrect_final_answer"


@dataclass(frozen=True)
class FailureClassification:
    category: FailureCategory
    evidence: tuple[str, ...]


def classify_failure(task: BenchmarkTask, result: HarnessResult) -> FailureCategory:
    """Classify with a fixed precedence; use the companion function for evidence."""
    return classify_failure_with_evidence(task, result).category


def classify_failure_with_evidence(
    task: BenchmarkTask, result: HarnessResult
) -> FailureClassification:
    """Return the highest-precedence symptom and reconstructable trace evidence.

    Precedence is: loop, repeated action, retry exhaustion, malformed output,
    invalid arguments, tool failure, wrong tool, context/max-step failure,
    premature termination, and incorrect final answer.
    """
    events = list(result.trace)

    def matching_failure(name: str) -> list[object]:
        return [event for event in events if event.payload.get("failure") == name]

    checks: list[tuple[FailureCategory, list[object]]] = [
        (FailureCategory.LOOP, matching_failure("loop_detected")),
        (FailureCategory.REPEATED_ACTION, matching_failure("repeated_action")),
        (
            FailureCategory.RETRY_EXHAUSTION,
            [
                event
                for event in events
                if event.type is TraceEventType.RETRY
                and event.payload.get("disposition") == "terminate"
                and "budget exhausted" in str(event.payload.get("reason", ""))
            ],
        ),
        (FailureCategory.MALFORMED_OUTPUT, matching_failure("malformed_model_output")),
        (FailureCategory.WRONG_TOOL, matching_failure("unknown_tool")),
        (FailureCategory.INVALID_ARGUMENTS, matching_failure("invalid_tool_arguments")),
        (FailureCategory.TOOL_FAILURE, matching_failure("tool_exception")),
    ]
    for category, evidence_events in checks:
        if evidence_events:
            return FailureClassification(
                category,
                tuple(f"event {event.sequence}: {event.payload}" for event in evidence_events),
            )

    metrics = evaluate_run(task, result)
    if not metrics.correct_tool_choice:
        actual = [
            event.payload.get("name") for event in events if event.type is TraceEventType.TOOL_CALL
        ]
        return FailureClassification(
            FailureCategory.WRONG_TOOL,
            (f"expected {[call.name for call in task.expected_tool_calls]}, got {actual}",),
        )
    if not metrics.correct_tool_arguments:
        actual = [
            event.payload.get("arguments")
            for event in events
            if event.type is TraceEventType.TOOL_CALL
        ]
        return FailureClassification(
            FailureCategory.INVALID_ARGUMENTS,
            (f"expected {[call.arguments for call in task.expected_tool_calls]}, got {actual}",),
        )
    if result.status is RunStatus.MAX_STEPS:
        return FailureClassification(
            FailureCategory.CONTEXT_FAILURE,
            ("run reached the global model-step limit",),
        )
    if result.status is RunStatus.FAILED and result.final_answer is None:
        return FailureClassification(
            FailureCategory.PREMATURE_TERMINATION,
            ("run failed without a final answer",),
        )
    return FailureClassification(
        FailureCategory.INCORRECT_FINAL_ANSWER,
        (f"expected {task.expected_answer!r}, got {result.final_answer!r}",),
    )
