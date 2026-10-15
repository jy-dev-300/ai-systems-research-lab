import pytest

from backend.application.baseline_runner import TemporaryBaselineRunner
from harnesses.contracts import RunStatus


@pytest.mark.anyio
async def test_temporary_baseline_keeps_demo_runnable(
    mock_backend, tool_registry, addition_task
) -> None:
    result = await TemporaryBaselineRunner(mock_backend, tool_registry).run(addition_task)

    assert result.status is RunStatus.SUCCEEDED
    assert result.final_answer == "42"
    assert result.model_calls == 2
    assert result.tool_calls == 1
    assert [event.sequence for event in result.trace] == list(range(len(result.trace)))
    assert [event.type.value for event in result.trace] == [
        "model_call",
        "tool_call",
        "tool_result",
        "model_call",
        "termination",
    ]
