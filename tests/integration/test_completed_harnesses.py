import pytest

from benchmarks.deterministic import DETERMINISTIC_SUITE
from harnesses.contracts import RunStatus
from harnesses.planner_executor import PlannerExecutorHarness
from harnesses.retry import RetryPolicy
from harnesses.simple_tool_loop import SimpleToolLoop
from harnesses.stateful_retry import StatefulRetryHarness
from serving.mock import DeterministicMockBackend
from tools.registry import default_tool_registry
from tracing.recorder import TraceRecorder


@pytest.mark.anyio
async def test_all_harnesses_complete_the_versioned_deterministic_suite() -> None:
    factories = (
        lambda task: SimpleToolLoop(
            DeterministicMockBackend(),
            default_tool_registry(),
            TraceRecorder(f"simple-{task.id}", task.id),
        ),
        lambda task: PlannerExecutorHarness(
            DeterministicMockBackend(),
            default_tool_registry(),
            TraceRecorder(f"planner-{task.id}", task.id),
        ),
        lambda task: StatefulRetryHarness(
            DeterministicMockBackend(),
            default_tool_registry(),
            TraceRecorder(f"retry-{task.id}", task.id),
            RetryPolicy(),
        ),
    )
    for task in DETERMINISTIC_SUITE.tasks:
        for build_harness in factories:
            result = await build_harness(task).run(task)
            assert result.status is RunStatus.SUCCEEDED
            assert result.final_answer == task.expected_answer
            assert result.model_calls == 2
            assert result.tool_calls == 1
