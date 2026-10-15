import pytest

from serving.models import ChatMessage, ModelRequest


@pytest.mark.anyio
async def test_mock_requests_tool_then_finishes(mock_backend, tool_registry, addition_task) -> None:
    metadata = {
        "expected_tool_calls": [call.model_dump() for call in addition_task.expected_tool_calls],
        "expected_answer": addition_task.expected_answer,
    }
    first = await mock_backend.generate(
        ModelRequest(
            messages=[ChatMessage(role="user", content=addition_task.prompt)],
            tools=tool_registry.definitions(),
            metadata=metadata,
        )
    )
    assert first.action.kind == "tool"
    assert first.action.tool_call is not None
    assert first.action.tool_call.name == "arithmetic"

    second = await mock_backend.generate(
        ModelRequest(
            messages=[
                ChatMessage(role="user", content=addition_task.prompt),
                ChatMessage(role="tool", content="42", name="arithmetic"),
            ],
            tools=tool_registry.definitions(),
            metadata=metadata,
        )
    )
    assert second.action.kind == "final"
    assert second.action.final_answer == "42"
