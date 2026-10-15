from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.adapters.knowledge import AdapterCapability, ProjectKnowledge
from harness.models import ExperimentConfig, Metric
from harnesses.contracts import Harness, HarnessResult


@dataclass
class AdapterRunOutput:
    status: str = "completed"
    value: dict[str, Any] = field(default_factory=dict)
    harness_result: HarnessResult | None = None
    metrics: dict[str, float | int | bool | None] = field(default_factory=dict)
    artifact_payloads: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    traces: list[dict[str, Any]] = field(default_factory=list)
    duration_ms: float | None = None


@dataclass
class AdapterEvaluation:
    success: bool
    quality: float | None = None
    metrics: list[Metric] = field(default_factory=list)
    domain_metrics: dict[str, float | int | bool | None] = field(default_factory=dict)
    failure_subtype: str | None = None
    evidence: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    latency_ms: float | None = None
    cost_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    dataset: str | None = None
    dataset_version: str | None = None


class TaskAdapter(ABC):
    """Owns domain behavior while the runner owns orchestration."""

    @abstractmethod
    def load_tasks(self, config: ExperimentConfig) -> list[dict[str, Any]]: ...

    @abstractmethod
    def setup(self, task: dict[str, Any], config: ExperimentConfig) -> None: ...

    @abstractmethod
    async def run(
        self, task: dict[str, Any], harness: Harness, config: ExperimentConfig
    ) -> AdapterRunOutput: ...

    @abstractmethod
    def evaluate(
        self, task: dict[str, Any], run_output: AdapterRunOutput, config: ExperimentConfig
    ) -> AdapterEvaluation: ...

    def artifacts(
        self,
        task: dict[str, Any],
        run_output: AdapterRunOutput,
        evaluation: AdapterEvaluation,
    ) -> dict[str, Any]:
        return run_output.artifact_payloads

    def classify_failure(
        self,
        task: dict[str, Any],
        run_output: AdapterRunOutput,
        evaluation: AdapterEvaluation,
    ) -> tuple[str | None, list[str]]:
        return evaluation.failure_subtype, evaluation.evidence

    def cleanup(self) -> None:
        return None

    def bind_repo(self, repo: str | Path) -> TaskAdapter:
        """Bind this adapter instance to one checkout/worktree for its lifetime."""
        resolved = Path(repo).resolve()
        current = getattr(self, "_assigned_repo", None)
        if current is not None and current != resolved:
            raise RuntimeError(
                f"Adapter is already bound to {current}; cannot rebind to {resolved}"
            )
        self._assigned_repo = resolved
        return self

    @property
    def assigned_repo(self) -> Path | None:
        return getattr(self, "_assigned_repo", None)

    def project_knowledge(self) -> ProjectKnowledge:
        """Return minimal generic knowledge for test/future adapters.

        Production adapters override this with project-specific mechanics.
        """
        return ProjectKnowledge(
            project_id=type(self).__name__.removesuffix("Adapter").casefold(),
            project_name=type(self).__name__.removesuffix("Adapter"),
            project_type="unspecified",
            repo=str(self.assigned_repo or "unbound"),
            summary="Adapter supports task execution and evaluation.",
            capabilities=frozenset(
                {
                    AdapterCapability.RUN,
                    AdapterCapability.EVALUATE,
                    AdapterCapability.OBSERVE,
                }
            ),
        )
