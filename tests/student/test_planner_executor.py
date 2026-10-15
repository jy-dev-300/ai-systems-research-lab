import pytest

from benchmarks.models import BenchmarkTask, ExpectedToolCall
from harnesses.contracts import HarnessConfig, RunStatus
from harnesses.planner_executor import PlannerExecutorHarness
from serving.models import ModelAction, ModelResponse, ToolCall
from tracing.events import TraceEventType
from tracing.recorder import TraceRecorder

pytestmark = pytest.mark.student


class ScriptedBackend:
    name = "scripted"

    def __init__(self, actions):
        self.actions = iter(actions)

    async def generate(self, request):
        return ModelResponse(action=next(self.actions))


@pytest.mark.anyio
async def test_planner_completes(mock_backend, tool_registry, addition_task) -> None:
    harness = PlannerExecutorHarness(
        mock_backend,
        tool_registry,
        TraceRecorder("planner", addition_task.id),
    )
    result = await harness.run(addition_task)
    assert result.status is RunStatus.SUCCEEDED
    assert result.final_answer == "42"
    assert any(event.payload.get("phase") == "planning" for event in result.trace)
    assert any(event.payload.get("phase") == "evaluation" for event in result.trace)


@pytest.mark.anyio
async def test_planner_handles_failed_execution(tool_registry, addition_task) -> None:
    backend = ScriptedBackend(
        [
            ModelAction(
                kind="tool",
                tool_call=ToolCall(id="bad", name="missing", arguments={}),
            )
        ]
    )
    result = await PlannerExecutorHarness(
        backend,
        tool_registry,
        TraceRecorder("planner-fail", addition_task.id),
    ).run(addition_task)
    assert result.status is RunStatus.FAILED
    assert any(event.type is TraceEventType.ERROR for event in result.trace)


@pytest.mark.anyio
async def test_planner_replans_for_additional_action(tool_registry, addition_task) -> None:
    call = lambda identifier: ModelAction(  # noqa: E731
        kind="tool",
        tool_call=ToolCall(
            id=identifier,
            name="arithmetic",
            arguments={"operation": "add", "left": 1, "right": 1},
        ),
    )
    backend = ScriptedBackend(
        [call("one"), call("two"), ModelAction(kind="final", final_answer="2")]
    )
    result = await PlannerExecutorHarness(
        backend,
        tool_registry,
        TraceRecorder("planner-replan", addition_task.id),
        HarnessConfig(max_steps=4),
    ).run(addition_task)
    assert result.status is RunStatus.SUCCEEDED
    assert any(event.payload.get("phase") == "replanning" for event in result.trace)


@pytest.mark.anyio
async def test_planner_accepts_empty_plan(mock_backend, tool_registry) -> None:
    task = BenchmarkTask(
        id="empty", prompt="Say done", expected_answer="done", expected_tool_calls=[]
    )
    result = await PlannerExecutorHarness(
        mock_backend,
        tool_registry,
        TraceRecorder("planner-empty", task.id),
    ).run(task)
    assert result.status is RunStatus.SUCCEEDED
    assert result.tool_calls == 0


@pytest.mark.anyio
async def test_planner_rejects_premature_fixture_completion(tool_registry) -> None:
    class PrematureBackend(ScriptedBackend):
        name = "deterministic-mock"

    task = BenchmarkTask(
        id="premature",
        prompt="Add",
        expected_answer="2",
        expected_tool_calls=[
            ExpectedToolCall(
                name="arithmetic",
                arguments={"operation": "add", "left": 1, "right": 1},
            )
        ],
    )
    backend = PrematureBackend([ModelAction(kind="final", final_answer="2")])
    result = await PlannerExecutorHarness(
        backend,
        tool_registry,
        TraceRecorder("planner-premature", task.id),
    ).run(task)
    assert result.status is RunStatus.FAILED
