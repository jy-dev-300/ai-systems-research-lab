from __future__ import annotations

from pydantic import BaseModel

from benchmarks.models import BenchmarkTask
from harnesses.contracts import HarnessResult, RunStatus
from tracing.events import TraceEventType


class HarnessMetrics(BaseModel):
    task_success: bool
    correct_tool_choice: bool
    correct_tool_arguments: bool
    unnecessary_tool_calls: int
    total_model_calls: int
    retries: int
    loops: int
    completion_correct: bool


def evaluate_run(task: BenchmarkTask, result: HarnessResult) -> HarnessMetrics:
    """Derive separate quality metrics from one trajectory."""
    calls = [event.payload for event in result.trace if event.type is TraceEventType.TOOL_CALL]
    expected = task.expected_tool_calls
    correct_names = len(calls) == len(expected) and all(
        actual.get("name") == wanted.name for actual, wanted in zip(calls, expected, strict=True)
    )
    correct_arguments = len(calls) == len(expected) and all(
        actual.get("arguments") == wanted.arguments
        for actual, wanted in zip(calls, expected, strict=True)
    )
    matched_calls = sum(
        actual.get("name") == wanted.name and actual.get("arguments") == wanted.arguments
        for actual, wanted in zip(calls, expected, strict=False)
    )
    retries = sum(event.type is TraceEventType.RETRY for event in result.trace)
    loops = sum(
        event.type in {TraceEventType.ERROR, TraceEventType.RETRY}
        and event.payload.get("failure") in {"loop_detected", "repeated_action"}
        for event in result.trace
    )
    completion_correct = (
        result.status is RunStatus.SUCCEEDED and result.final_answer == task.expected_answer
    )
    return HarnessMetrics(
        task_success=completion_correct and correct_names and correct_arguments,
        correct_tool_choice=correct_names,
        correct_tool_arguments=correct_arguments,
        unnecessary_tool_calls=max(0, len(calls) - matched_calls),
        total_model_calls=result.model_calls,
        retries=retries,
        loops=loops,
        completion_correct=completion_correct,
    )
