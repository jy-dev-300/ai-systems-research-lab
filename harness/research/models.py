from __future__ import annotations

from datetime import date
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field


class EvidenceType(StrEnum):
    PAPER = "paper"
    IMPLEMENTATION = "implementation"
    ENGINEERING_REPORT = "engineering_report"
    INTERNAL_EXPERIMENT = "internal_experiment"


class ResearchDocument(BaseModel):
    id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    publication_date: date | None = None
    venue: str | None = None
    abstract: str = ""
    text: str = ""
    url: str | None = None
    implementation_url: str | None = None
    tags: list[str] = Field(default_factory=list)
    provenance: str
    evidence_type: EvidenceType


class ResearchEvidence(BaseModel):
    document_id: str
    title: str
    url: str | None = None
    evidence_type: EvidenceType
    relevance_score: float = Field(ge=0)
    claim_scope: str = "external"


class CandidateExperiment(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    observed_bottleneck: str
    candidate_technique: str
    supporting_sources: list[ResearchEvidence] = Field(default_factory=list)
    mechanism: str
    expected_metric_affected: list[str]
    assumptions: list[str] = Field(default_factory=list)
    controlled_experiment: str
    implementation_reference: str | None = None
    evidence_level: str
    confidence: float = Field(ge=0, le=1)
    internal_verification_status: str = "unverified"


class ResearchHypothesis(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    statement: str
    mechanism: str
    expected_metrics: list[str]
    source_candidate_id: str | None = None
    status: str = "proposed"


class ExperimentPlan(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    hypothesis_id: str
    control_config: str
    treatment_config: str
    verification_rules: list[str]
    status: str = "proposed"


class VerificationResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    hypothesis_id: str
    experiment_plan_id: str
    build_task_id: str
    build_result_id: str
    experiment_run_ids: list[str]
    supported: bool | None = None
    evidence: list[str] = Field(default_factory=list)


class ResearchFinding(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    hypothesis_id: str
    verification_result_id: str
    decision: str
    claim: str
    evidence_scope: str = "internal_experiment"
