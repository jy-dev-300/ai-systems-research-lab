from benchmarks.models import BenchmarkTask, ExpectedToolCall
from evaluation.failures import FailureCategory, classify_failure_with_evidence
from evaluation.metrics import evaluate_run
from harnesses.contracts import HarnessResult, RunStatus
from tracing.events import TraceEvent, TraceEventType


def event(sequence, kind, payload):
    return TraceEvent(run_id="run", task_id="task", sequence=sequence, type=kind, payload=payload)


def test_metrics_keep_quality_dimensions_separate(addition_task) -> None:
    result = HarnessResult(
        addition_task.id,
        RunStatus.SUCCEEDED,
        "42",
        2,
        1,
        (
            event(
                0,
                TraceEventType.TOOL_CALL,
                {
                    "name": "arithmetic",
                    "arguments": {"operation": "add", "left": 17, "right": 25},
                },
            ),
        ),
    )
    metrics = evaluate_run(addition_task, result)
    assert metrics.task_success
    assert metrics.correct_tool_choice
    assert metrics.correct_tool_arguments
    assert metrics.unnecessary_tool_calls == 0


def test_failure_precedence_returns_loop_evidence() -> None:
    task = BenchmarkTask(
        id="task",
        prompt="add",
        expected_answer="2",
        expected_tool_calls=[ExpectedToolCall(name="arithmetic", arguments={})],
    )
    result = HarnessResult(
        "task",
        RunStatus.FAILED,
        None,
        3,
        1,
        (
            event(0, TraceEventType.ERROR, {"failure": "invalid_tool_arguments"}),
            event(1, TraceEventType.RETRY, {"failure": "loop_detected"}),
        ),
    )
    classification = classify_failure_with_evidence(task, result)
    assert classification.category is FailureCategory.LOOP
    assert "event 1" in classification.evidence[0]
