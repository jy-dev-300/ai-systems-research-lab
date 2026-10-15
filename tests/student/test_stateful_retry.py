import pytest

from harnesses.contracts import HarnessConfig, RunStatus
from harnesses.retry import RetryPolicy
from harnesses.stateful_retry import StatefulRetryHarness
from serving.models import ModelAction, ModelResponse, ToolCall
from tracing.events import TraceEventType
from tracing.recorder import TraceRecorder

pytestmark = pytest.mark.student


class ScriptedBackend:
    name = "retry-script"

    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)

    async def generate(self, request):
        outcome = next(self.outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return ModelResponse(action=outcome)


@pytest.mark.anyio
async def test_timeout_retries_then_finishes(tool_registry, addition_task) -> None:
    backend = ScriptedBackend([TimeoutError("slow"), ModelAction(kind="final", final_answer="42")])
    result = await StatefulRetryHarness(
        backend,
        tool_registry,
        TraceRecorder("retry-timeout", addition_task.id),
        RetryPolicy(),
        HarnessConfig(max_steps=3, max_retries=1),
    ).run(addition_task)
    assert result.status is RunStatus.SUCCEEDED
    assert result.model_calls == 2
    assert any(event.type is TraceEventType.RETRY for event in result.trace)


@pytest.mark.anyio
async def test_invalid_arguments_fall_back_then_finish(tool_registry, addition_task) -> None:
    backend = ScriptedBackend(
        [
            ModelAction(
                kind="tool",
                tool_call=ToolCall(id="bad", name="arithmetic", arguments={"left": 1}),
            ),
            ModelAction(kind="final", final_answer="42"),
        ]
    )
    result = await StatefulRetryHarness(
        backend,
        tool_registry,
        TraceRecorder("retry-args", addition_task.id),
        RetryPolicy(),
        HarnessConfig(max_steps=3, max_retries=1),
    ).run(addition_task)
    assert result.status is RunStatus.SUCCEEDED
    assert result.tool_calls == 1


@pytest.mark.anyio
async def test_repeated_action_terminates_as_loop(tool_registry, addition_task) -> None:
    action = ModelAction(
        kind="tool",
        tool_call=ToolCall(
            id="repeat",
            name="arithmetic",
            arguments={"operation": "add", "left": 1, "right": 1},
        ),
    )
    result = await StatefulRetryHarness(
        ScriptedBackend([action, action, action]),
        tool_registry,
        TraceRecorder("retry-loop", addition_task.id),
        RetryPolicy(),
        HarnessConfig(max_steps=3, max_retries=2),
    ).run(addition_task)
    assert result.status is RunStatus.FAILED
    assert any(event.payload.get("failure") == "loop_detected" for event in result.trace)


@pytest.mark.anyio
async def test_malformed_output_exhausts_zero_retry_budget(tool_registry, addition_task) -> None:
    result = await StatefulRetryHarness(
        ScriptedBackend([ValueError("bad structured output")]),
        tool_registry,
        TraceRecorder("retry-malformed", addition_task.id),
        RetryPolicy(),
        HarnessConfig(max_steps=1, max_retries=0),
    ).run(addition_task)
    assert result.status is RunStatus.FAILED
    assert any(event.payload.get("failure") == "malformed_model_output" for event in result.trace)


@pytest.mark.anyio
async def test_tool_exception_uses_fallback_policy(tool_registry, addition_task) -> None:
    backend = ScriptedBackend(
        [
            ModelAction(
                kind="tool",
                tool_call=ToolCall(
                    id="missing-sku",
                    name="inventory_lookup",
                    arguments={"sku": "NOT-THERE"},
                ),
            ),
            ModelAction(kind="final", final_answer="42"),
        ]
    )
    result = await StatefulRetryHarness(
        backend,
        tool_registry,
        TraceRecorder("retry-tool", addition_task.id),
        RetryPolicy(),
        HarnessConfig(max_steps=2, max_retries=1),
    ).run(addition_task)
    assert result.status is RunStatus.SUCCEEDED
    assert any(event.payload.get("failure") == "tool_exception" for event in result.trace)
