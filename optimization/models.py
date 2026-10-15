from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class InferenceConfiguration(BaseModel):
    name: str
    backend: Literal["mock", "vllm", "sglang"]
    model: str
    quantization: str | None = None
    max_model_length: int
    concurrency: int = 1
    batch_size: int = 1
    prefix_cache_enabled: bool = False
    arrival_pattern: Literal["serial", "burst", "poisson"] = "serial"
    extra: dict[str, object] = Field(default_factory=dict)


class ExperimentResult(BaseModel):
    configuration: InferenceConfiguration
    task_success_rate: float
    p95_latency_ms: float
    throughput_requests_per_second: float
    peak_gpu_memory_mb: float | None
    cost_per_task: float | None
    failures: int = 0
