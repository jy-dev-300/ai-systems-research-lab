from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class AdapterCapability(StrEnum):
    INSPECT = "inspect"
    SETUP = "setup"
    BUILD = "build"
    RUN = "run"
    TEST = "test"
    EVALUATE = "evaluate"
    EXPERIMENT = "experiment"
    OBSERVE = "observe"


class MetricDirection(StrEnum):
    HIGHER = "higher"
    LOWER = "lower"
    TARGET = "target"


class CommandKnowledge(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    command: tuple[str, ...]
    purpose: str
    timeout_seconds: float = Field(default=300, gt=0)


class MetricKnowledge(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    direction: MetricDirection
    primary: bool = False
    unit: str | None = None


class EvaluationKnowledge(BaseModel):
    model_config = ConfigDict(frozen=True)

    operation: str
    dataset: str | None = None
    dataset_version: str | None = None
    split: str | None = None
    protected: bool = True
    metrics: tuple[MetricKnowledge, ...] = ()


class ProjectKnowledge(BaseModel):
    """Concise operational manual for one adapter bound to one checkout."""

    model_config = ConfigDict(frozen=True)

    project_id: str
    project_name: str
    project_type: str
    repo: str
    summary: str
    capabilities: frozenset[AdapterCapability]
    editable_paths: tuple[str, ...] = ()
    protected_paths: tuple[str, ...] = ()
    entry_points: tuple[str, ...] = ()
    setup_commands: tuple[CommandKnowledge, ...] = ()
    run_commands: tuple[CommandKnowledge, ...] = ()
    test_commands: tuple[CommandKnowledge, ...] = ()
    evaluation: EvaluationKnowledge | None = None
    baseline: str | None = None
    configurable_components: tuple[str, ...] = ()
    reproducibility_requirements: tuple[str, ...] = ()
    artifacts_to_preserve: tuple[str, ...] = ()
    integrity_constraints: tuple[str, ...] = ()
    environment_assumptions: tuple[str, ...] = ()

    def classification_context(self) -> dict[str, object]:
        """Return the small context needed for top-level workflow classification."""
        return {
            "project_type": self.project_type,
            "summary": self.summary,
            "available_capabilities": sorted(item.value for item in self.capabilities),
            "metrics": [item.name for item in self.evaluation.metrics] if self.evaluation else [],
            "configurable_components": list(self.configurable_components),
            "protected_surfaces": list(self.protected_paths),
        }
