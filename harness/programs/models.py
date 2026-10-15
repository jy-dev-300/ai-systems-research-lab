from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Workflow(StrEnum):
    BUILD = "BUILD"
    EVAL = "EVAL"
    RESEARCH = "RESEARCH"


class ProgramStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    REVIEW_REQUIRED = "review_required"


class ProgramDefinition(BaseModel):
    project: str
    adapter: str
    repo: str
    evaluation_config: str
    aliases: tuple[str, ...] = ()


class ProgramDocument(BaseModel):
    path: str
    filename: str
    contents: str
    sha256: str

    @classmethod
    def load(cls, path: str | Path) -> ProgramDocument:
        source = Path(path).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Program not found: {source}")
        contents = source.read_text(encoding="utf-8")
        if not contents.strip():
            raise ValueError("Program cannot be empty")
        return cls(
            path=str(source),
            filename=source.name,
            contents=contents,
            sha256=hashlib.sha256(contents.encode("utf-8")).hexdigest(),
        )


class ProjectState(BaseModel):
    repo: str
    exists: bool
    git_commit: str | None = None
    dirty: bool | None = None
    summary: str


class WorkflowContext(BaseModel):
    program: ProgramDocument
    project: str
    adapter: str
    capabilities: tuple[str, ...]
    project_state: ProjectState
    project_knowledge: dict[str, Any] = Field(default_factory=dict)


class WorkflowDecision(BaseModel):
    workflow: Workflow
    confidence: float = Field(ge=0, le=1)
    reason: str
    required_capabilities: tuple[str, ...]
    classifier: str
    classifier_version: str
    model: str
    prompt_version: str
    probabilities: dict[str, float] = Field(default_factory=dict)
    latency_ms: float = Field(default=0, ge=0)


class ProgramRunResult(BaseModel):
    program_run_id: str = Field(default_factory=lambda: str(uuid4()))
    status: ProgramStatus
    program: ProgramDocument
    project: str | None = None
    adapter: str | None = None
    target_repo_commit: str | None = None
    worktree: str | None = None
    branch: str | None = None
    decision: WorkflowDecision | None = None
    experiment_run_ids: tuple[str, ...] = ()
    build_id: str | None = None
    candidate_experiment_ids: tuple[str, ...] = ()
    diff_artifact: str | None = None
    artifact_dir: str | None = None
    summary: str
    errors: tuple[str, ...] = ()
    outputs: dict[str, Any] = Field(default_factory=dict)
    project_knowledge: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
