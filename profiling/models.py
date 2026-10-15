from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class RequestStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


class RequestProfile(BaseModel):
    request_id: str
    status: RequestStatus
    time_to_first_token_ms: float | None
    total_latency_ms: float
    queue_time_ms: float | None
    input_tokens: int
    output_tokens: int
    output_tokens_per_second: float | None
    memory_before_mb: float | None
    memory_after_mb: float | None
    cache_hit: bool | None = None
    generation_duration_ms: float | None = None
    error: str | None = None


class BottleneckBreakdown(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_inference_ms: float
    queueing_ms: float
    tool_execution_ms: float
    retry_ms: float
    repeated_model_call_ms: float
    harness_overhead_ms: float
