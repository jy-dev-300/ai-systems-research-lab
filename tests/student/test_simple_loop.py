import pytest

from harnesses.contracts import HarnessConfig, RunStatus
from harnesses.simple_tool_loop import SimpleToolLoop
from tracing.recorder import TraceRecorder

pytestmark = pytest.mark.student


@pytest.mark.anyio
async def test_simple_loop_completes_tool_task(mock_backend, tool_registry, addition_task) -> None:
    harness = SimpleToolLoop(
        backend=mock_backend,
        tools=tool_registry,
        tracer=TraceRecorder("student-run", addition_task.id),
        config=HarnessConfig(max_steps=4),
    )
    result = await harness.run(addition_task)
    assert result.status is RunStatus.SUCCEEDED
    assert result.final_answer == "42"
    assert result.model_calls == 2
    assert result.tool_calls == 1


class NeverFinalBackend:
    name = "never-final"

    async def health(self) -> bool:
        return True

    async def generate(self, request):
        from serving.models import ModelAction, ModelResponse, ToolCall

        return ModelResponse(
            action=ModelAction(
                kind="tool",
                tool_call=ToolCall(
                    id="repeat",
                    name="arithmetic",
                    arguments={"operation": "add", "left": 1, "right": 1},
                ),
            )
        )


@pytest.mark.anyio
async def test_simple_loop_enforces_max_steps(tool_registry, addition_task) -> None:
    harness = SimpleToolLoop(
        backend=NeverFinalBackend(),
        tools=tool_registry,
        tracer=TraceRecorder("bounded-run", addition_task.id),
        config=HarnessConfig(max_steps=2),
    )
    result = await harness.run(addition_task)
    assert result.status is RunStatus.MAX_STEPS
    assert result.model_calls == 2


@pytest.mark.anyio
async def test_simple_loop_returns_failed_result_for_backend_error(
    tool_registry, addition_task
) -> None:
    class BrokenBackend:
        name = "broken"

        async def generate(self, request):
            raise RuntimeError("offline")

    result = await SimpleToolLoop(
        BrokenBackend(),
        tool_registry,
        TraceRecorder("broken-run", addition_task.id),
    ).run(addition_task)
    assert result.status is RunStatus.FAILED
    assert result.model_calls == 1
    assert result.trace[-1].type.value == "termination"
