from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Metric(BaseModel):
    name: str
    value: float
    unit: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Artifact(BaseModel):
    name: str
    path: str
    media_type: str = "application/octet-stream"
    metadata: dict[str, Any] = Field(default_factory=dict)


class FailureType(StrEnum):
    REASONING = "reasoning_failure"
    TOOL = "tool_failure"
    RETRIEVAL = "retrieval_failure"
    PLANNING = "planning_failure"
    EXECUTION = "execution_failure"
    TIMEOUT = "timeout"
    RESOURCE = "resource_failure"
    EVALUATION = "evaluation_failure"
    MODEL = "model_failure"
    CONFIGURATION = "configuration_failure"
    UNKNOWN = "unknown"


class FailureClassification(BaseModel):
    type: FailureType
    subtype: str | None = None
    evidence: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)


class TraceEvent(BaseModel):
    sequence: int = Field(ge=0)
    type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    duration_ms: float | None = Field(default=None, ge=0)
    payload: dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    provider: str = "mock"
    name: str = "deterministic-mock"
    backend: str = "deterministic"
    base_url: str | None = None
    api_key_env: str | None = None


class HarnessSpec(BaseModel):
    type: str = "simple_loop"
    max_steps: int = Field(default=6, ge=1)
    max_retries: int = Field(default=2, ge=0)
    parameters: dict[str, Any] = Field(default_factory=dict)


class AdapterSpec(BaseModel):
    name: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class EvaluationSpec(BaseModel):
    dataset: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class TrackingSpec(BaseModel):
    type: Literal["local", "mlflow"] = "local"
    uri: str = "./mlruns"
    experiment: str | None = None


class ExperimentIdentity(BaseModel):
    name: str
    domain: str
    description: str | None = None


class ExperimentConfig(BaseModel):
    config_version: str = "1.0"
    experiment: ExperimentIdentity
    harness: HarnessSpec = Field(default_factory=HarnessSpec)
    model: ModelConfig = Field(default_factory=ModelConfig)
    adapter: AdapterSpec
    evaluation: EvaluationSpec = Field(default_factory=EvaluationSpec)
    tracking: TrackingSpec = Field(default_factory=TrackingSpec)
    parameters: dict[str, Any] = Field(default_factory=dict)
    seed: int = 0
    output_dir: str = ".harness/runs"

    @model_validator(mode="after")
    def domain_matches_adapter(self) -> ExperimentConfig:
        if not self.experiment.domain.strip():
            raise ValueError("experiment.domain cannot be empty")
        return self

    @classmethod
    def from_file(cls, path: str | Path) -> ExperimentConfig:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Experiment config not found: {source}")
        text = source.read_text(encoding="utf-8")
        data = json.loads(text) if source.suffix.lower() == ".json" else yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("Experiment config must contain a mapping")
        return cls.model_validate(data)

    def fingerprint(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


class ExperimentResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    experiment_id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    build_task_id: str | None = None
    build_result_id: str | None = None
    research_hypothesis_id: str | None = None
    experiment_plan_id: str | None = None
    task_id: str
    domain: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    model: str
    provider: str
    backend: str
    harness_type: str
    git_commit: str | None = None
    config_version: str
    config_fingerprint: str
    config: dict[str, Any]
    success: bool
    quality: float | None = None
    latency_ms: float = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0, ge=0)
    model_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    retries: int = Field(default=0, ge=0)
    failure: FailureClassification | None = None
    metrics: list[Metric] = Field(default_factory=list)
    domain_metrics: dict[str, float | int | bool | None] = Field(default_factory=dict)
    artifacts: list[Artifact] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

    def metric_values(self) -> dict[str, float]:
        values = {metric.name: metric.value for metric in self.metrics}
        values.update(
            {
                "success": float(self.success),
                "latency_ms": self.latency_ms,
                "tokens": float(self.input_tokens + self.output_tokens),
                "cost_usd": self.cost_usd,
                "model_calls": float(self.model_calls),
                "tool_calls": float(self.tool_calls),
                "retries": float(self.retries),
            }
        )
        if self.quality is not None:
            values["quality"] = self.quality
        for key, value in self.domain_metrics.items():
            if isinstance(value, (int, float, bool)):
                values[f"domain.{key}"] = float(value)
        return values
