from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    tool_call_id: str | None = None


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class ModelRequest(BaseModel):
    messages: list[ChatMessage]
    tools: list[ToolDefinition] = Field(default_factory=list)
    temperature: float = 0.0
    max_output_tokens: int = 256
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]


class ModelAction(BaseModel):
    kind: Literal["tool", "final"]
    tool_call: ToolCall | None = None
    final_answer: str | None = None

    @model_validator(mode="after")
    def require_matching_payload(self) -> ModelAction:
        if self.kind == "tool" and self.tool_call is None:
            raise ValueError("tool action requires tool_call")
        if self.kind == "final" and self.final_answer is None:
            raise ValueError("final action requires final_answer")
        return self


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class InferenceMetrics(BaseModel):
    total_latency_ms: float = 0.0
    time_to_first_token_ms: float | None = None
    queue_time_ms: float | None = None
    cache_hit: bool | None = None
    gpu_memory_before_mb: float | None = None
    gpu_memory_after_mb: float | None = None


class ModelResponse(BaseModel):
    action: ModelAction
    usage: Usage = Field(default_factory=Usage)
    metrics: InferenceMetrics = Field(default_factory=InferenceMetrics)
    raw: dict[str, Any] = Field(default_factory=dict)
