from __future__ import annotations

from serving.models import (
    InferenceMetrics,
    ModelAction,
    ModelRequest,
    ModelResponse,
    ToolCall,
    Usage,
)


class DeterministicMockBackend:
    """A scripted backend for repeatable application tests, never performance claims."""

    name = "deterministic-mock"

    async def health(self) -> bool:
        return True

    async def generate(self, request: ModelRequest) -> ModelResponse:
        expected_calls = request.metadata.get("expected_tool_calls", [])
        expected_answer = str(request.metadata.get("expected_answer", ""))
        tool_messages = [message for message in request.messages if message.role == "tool"]

        if len(tool_messages) < len(expected_calls):
            expected = expected_calls[len(tool_messages)]
            action = ModelAction(
                kind="tool",
                tool_call=ToolCall(
                    id=f"mock-call-{len(tool_messages) + 1}",
                    name=expected["name"],
                    arguments=expected["arguments"],
                ),
            )
        else:
            action = ModelAction(kind="final", final_answer=expected_answer)

        input_tokens = sum(len(message.content.split()) for message in request.messages)
        output_tokens = 8 if action.kind == "tool" else max(1, len(expected_answer.split()))
        return ModelResponse(
            action=action,
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
            metrics=InferenceMetrics(total_latency_ms=1.0, time_to_first_token_ms=0.25),
            raw={"warning": "synthetic response and metrics"},
        )
