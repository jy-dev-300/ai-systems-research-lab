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

from harness.models import ExperimentResult, TraceEvent


class BuildApprovalState(StrEnum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    RUNNING = "running"
    REVIEW_REQUIRED = "review_required"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class BuildStatus(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    LIMIT_REACHED = "limit_reached"
    HUMAN_REVIEW_REQUIRED = "human_review_required"


class CriterionKind(StrEnum):
    TESTS_PASS = "tests_pass"
    EXPERIMENT_SUCCESS = "experiment_success"
    METRIC = "metric"
    FILE_EXISTS = "file_exists"
    DIFF_NONEMPTY = "diff_nonempty"
    HUMAN_REVIEW = "human_review"


class AcceptanceCriterion(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    description: str
    kind: CriterionKind
    metric_name: str | None = None
    operator: Literal[">=", ">", "<=", "<", "=="] = ">="
    threshold: float | None = None
    aggregation: Literal["mean", "min", "max"] = "mean"
    path: str | None = None

    @model_validator(mode="after")
    def validate_kind_fields(self) -> AcceptanceCriterion:
        if self.kind is CriterionKind.METRIC and (
            self.metric_name is None or self.threshold is None
        ):
            raise ValueError("metric criteria require metric_name and threshold")
        if self.kind is CriterionKind.FILE_EXISTS and self.path is None:
            raise ValueError("file_exists criteria require path")
        return self


class ValidationCommand(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    command: tuple[str, ...]
    timeout_seconds: float = Field(default=300, gt=0)

    @model_validator(mode="after")
    def command_is_not_empty(self) -> ValidationCommand:
        if not self.command:
            raise ValueError("validation command cannot be empty")
        return self


class BuildTask(BaseModel):
    """Immutable, human-approved build specification and fixed success criteria."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    version: str = "1.0"
    target_project: str
    adapter_name: str
    goal: str
    requirements: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    acceptance_criteria: tuple[AcceptanceCriterion, ...]
    relevant_files: tuple[str, ...] = ()
    allowed_actions: frozenset[str] = frozenset({"inspect", "edit", "test", "evaluate"})
    approved_high_risk_actions: frozenset[str] = frozenset()
    max_iterations: int = Field(default=3, ge=1)
    max_cost_usd: float | None = Field(default=None, gt=0)
    max_tokens: int | None = Field(default=None, gt=0)
    research_hypothesis_id: str | None = None
    experiment_plan_id: str | None = None
    validation_config_paths: tuple[str, ...] = ()
    validation_commands: tuple[ValidationCommand, ...] = ()
    approval_state: BuildApprovalState = BuildApprovalState.PROPOSED
    output_dir: str = ".harness/builds"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def criteria_are_unique(self) -> BuildTask:
        identifiers = [criterion.id for criterion in self.acceptance_criteria]
        if not identifiers:
            raise ValueError("BuildTask requires at least one acceptance criterion")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("acceptance criterion IDs must be unique")
        return self

    @classmethod
    def from_file(cls, path: str | Path) -> BuildTask:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Build task not found: {source}")
        text = source.read_text(encoding="utf-8")
        payload = json.loads(text) if source.suffix.lower() == ".json" else yaml.safe_load(text)
        if not isinstance(payload, dict):
            raise ValueError("Build task must contain a mapping")
        return cls.model_validate(payload)

    def spec_fingerprint(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


class RepositoryContext(BaseModel):
    root: str
    files: tuple[str, ...]
    relevant_contents: dict[str, str] = Field(default_factory=dict)
    git_commit: str | None = None
    dirty: bool | None = None
    project_knowledge: dict[str, Any] = Field(default_factory=dict)


class BuildPlan(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    summary: str
    steps: tuple[str, ...]
    expected_files: tuple[str, ...] = ()
    major_architecture_change: bool = False


class FileChange(BaseModel):
    path: str
    operation: Literal["create", "update", "delete"]
    content: str | None = None
    rationale: str = ""

    @model_validator(mode="after")
    def content_matches_operation(self) -> FileChange:
        if self.operation != "delete" and self.content is None:
            raise ValueError("create/update changes require content")
        return self


class BuildProposal(BaseModel):
    changes: tuple[FileChange, ...] = ()
    actions: frozenset[str] = frozenset({"edit"})
    summary: str
    requires_human_approval: bool = False
    tokens_used: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0, ge=0)
    latency_ms: float = Field(default=0, ge=0)


class AppliedChange(BaseModel):
    path: str
    operation: str
    before_sha256: str | None = None
    after_sha256: str | None = None
    diff: str = ""


class TestExecutionRecord(BaseModel):
    name: str
    command: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    duration_ms: float = Field(default=0, ge=0)


class AcceptanceCriterionResult(BaseModel):
    criterion_id: str
    passed: bool
    evidence: tuple[str, ...] = ()


class BuildValidation(BaseModel):
    criterion_results: tuple[AcceptanceCriterionResult, ...]
    tests: tuple[TestExecutionRecord, ...] = ()
    experiment_results: tuple[ExperimentResult, ...] = ()
    metrics: dict[str, float] = Field(default_factory=dict)
    failures: tuple[str, ...] = ()
    requires_human_review: bool = False

    @property
    def accepted(self) -> bool:
        return bool(self.criterion_results) and all(item.passed for item in self.criterion_results)


class BuildDiagnosis(BaseModel):
    summary: str
    evidence: tuple[str, ...] = ()
    next_actions: tuple[str, ...] = ()
    unrecoverable: bool = False


class BuildIteration(BaseModel):
    number: int
    proposal: BuildProposal
    applied_changes: tuple[AppliedChange, ...] = ()
    validation: BuildValidation | None = None
    diagnosis: BuildDiagnosis | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None


class BuildResult(BaseModel):
    build_id: str = Field(default_factory=lambda: str(uuid4()))
    task: BuildTask
    spec_fingerprint: str
    implementation_plan: BuildPlan | None = None
    files_changed: tuple[str, ...] = ()
    iterations: tuple[BuildIteration, ...] = ()
    tests_executed: tuple[TestExecutionRecord, ...] = ()
    evals_executed: tuple[str, ...] = ()
    experiment_results: tuple[ExperimentResult, ...] = ()
    metrics: dict[str, float] = Field(default_factory=dict)
    failures_encountered: tuple[str, ...] = ()
    fixes_attempted: tuple[str, ...] = ()
    traces: tuple[TraceEvent, ...] = ()
    final_status: BuildStatus
    approval_state: BuildApprovalState
    acceptance_criteria_results: tuple[AcceptanceCriterionResult, ...] = ()
    unresolved_issues: tuple[str, ...] = ()
    tokens_used: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0, ge=0)
    latency_ms: float = Field(default=0, ge=0)
    git_commit: str | None = None
    diff_artifact: str | None = None
    research_hypothesis_id: str | None = None
    experiment_plan_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
