from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TraceEventView(BaseModel):
    sequence: int
    type: str
    payload: dict[str, Any]


class DemoTaskResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    task_id: str
    status: str
    expected_answer: str
    final_answer: str | None
    correct: bool
    model_calls: int
    tool_calls: int
    trace: list[TraceEventView]


class DemoRunResponse(BaseModel):
    backend: str
    warning: str
    tasks: list[DemoTaskResult]


class LabMetricView(BaseModel):
    name: str
    value: float
    unit: str | None = None


class LabSystemView(BaseModel):
    id: str
    name: str
    project_type: str
    summary: str
    status: str
    repo: str
    repo_exists: bool
    capabilities: list[str]
    metrics: list[LabMetricView] = Field(default_factory=list)
    run_count: int = 0
    success_count: int = 0
    last_run_at: datetime | None = None


class LabOpportunityView(BaseModel):
    id: str
    system_id: str
    title: str
    description: str
    severity: str
    status: str
    run_id: str | None = None
    affected_metrics: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    candidate_experiment_id: str | None = None


class LabExperimentView(BaseModel):
    run_id: str
    task_id: str
    domain: str
    name: str
    timestamp: datetime
    success: bool
    is_mock: bool
    harness_type: str
    model: str
    latency_ms: float
    cost_usd: float
    metrics: list[LabMetricView] = Field(default_factory=list)
    failure: dict[str, Any] | None = None
    trace: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    output: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class ResearchDocumentView(BaseModel):
    id: str
    title: str
    authors: list[str]
    publication_date: str | None = None
    venue: str | None = None
    abstract: str
    url: str
    implementation_url: str | None = None
    tags: list[str] = Field(default_factory=list)
    evidence_type: str


class LabActivityView(BaseModel):
    id: str
    kind: str
    title: str
    status: str
    timestamp: datetime | None = None
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)


class LabOverviewResponse(BaseModel):
    generated_at: datetime
    artifact_root: str
    mlflow_url: str
    systems: list[LabSystemView]
    opportunities: list[LabOpportunityView]
    experiments: list[LabExperimentView]
    research_documents: list[ResearchDocumentView]
    activity: list[LabActivityView]
    counts: dict[str, int]
