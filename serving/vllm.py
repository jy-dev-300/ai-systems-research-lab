from __future__ import annotations

import json
from time import perf_counter

import httpx

from serving.models import (
    InferenceMetrics,
    ModelAction,
    ModelRequest,
    ModelResponse,
    ToolCall,
    Usage,
)


class VLLMBackend:
    """Thin adapter for a vLLM OpenAI-compatible server.

    Supports non-streaming structured tool responses. Trustworthy TTFT requires
    a streaming path, while queue, cache, and memory metrics require backend data.
    """

    name = "vllm"

    def __init__(self, base_url: str, model: str, timeout_seconds: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/models")
            return response.is_success
        except httpx.HTTPError:
            return False

    async def generate(self, request: ModelRequest) -> ModelResponse:
        payload = {
            "model": self.model,
            "messages": [message.model_dump(exclude_none=True) for message in request.messages],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in request.tools
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
        }
        started = perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(f"{self.base_url}/chat/completions", json=payload)
            response.raise_for_status()
        elapsed_ms = (perf_counter() - started) * 1000
        body = response.json()
        choice = body["choices"][0]["message"]
        tool_calls = choice.get("tool_calls") or []
        if tool_calls:
            call = tool_calls[0]
            action = ModelAction(
                kind="tool",
                tool_call=ToolCall(
                    id=call["id"],
                    name=call["function"]["name"],
                    arguments=json.loads(call["function"]["arguments"]),
                ),
            )
        else:
            action = ModelAction(kind="final", final_answer=choice.get("content") or "")
        usage = body.get("usage", {})
        return ModelResponse(
            action=action,
            usage=Usage(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            ),
            metrics=InferenceMetrics(total_latency_ms=elapsed_ms),
            raw=body,
        )
