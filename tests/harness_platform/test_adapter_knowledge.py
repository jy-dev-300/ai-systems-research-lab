from __future__ import annotations

from pathlib import Path

import pytest

from harness.adapters.knowledge import AdapterCapability, MetricDirection, ProjectKnowledge
from harness.adapters.registry import adapter_registry, register_builtin_adapters
from harness.build.adapter import ExperimentBuildAdapter
from harness.build.models import AcceptanceCriterion, BuildTask, CriterionKind


def test_builtin_adapter_knowledge_is_discoverable_and_serializable(tmp_path: Path) -> None:
    register_builtin_adapters()
    adapter = adapter_registry.create("bo", repo=tmp_path)

    knowledge = adapter.project_knowledge()

    assert knowledge.repo == str(tmp_path.resolve())
    assert {AdapterCapability.TEST, AdapterCapability.EVALUATE}.issubset(knowledge.capabilities)
    assert knowledge.evaluation is not None
    assert any(
        metric.direction is MetricDirection.HIGHER for metric in knowledge.evaluation.metrics
    )
    assert ProjectKnowledge.model_validate_json(knowledge.model_dump_json()) == knowledge


def test_adapter_instances_are_bound_without_shared_repo_state(tmp_path: Path) -> None:
    register_builtin_adapters()
    first_repo = tmp_path / "first"
    second_repo = tmp_path / "second"
    first_repo.mkdir()
    second_repo.mkdir()

    first = adapter_registry.create("swebench", repo=first_repo)
    second = adapter_registry.create("swebench", repo=second_repo)

    assert first is not second
    assert first.project_knowledge().repo == str(first_repo.resolve())
    assert second.project_knowledge().repo == str(second_repo.resolve())
    with pytest.raises(RuntimeError, match="already bound"):
        first.bind_repo(second_repo)


def test_optional_capabilities_do_not_claim_unavailable_build_system(tmp_path: Path) -> None:
    register_builtin_adapters()
    capabilities = (
        adapter_registry.create("myvoice", repo=tmp_path).project_knowledge().capabilities
    )

    assert AdapterCapability.EVALUATE in capabilities
    assert AdapterCapability.BUILD not in capabilities
    assert AdapterCapability.TEST not in capabilities


def test_build_inspection_includes_bound_adapter_knowledge(tmp_path: Path) -> None:
    task = BuildTask(
        target_project=str(tmp_path),
        adapter_name="harness",
        goal="Inspect the harness.",
        acceptance_criteria=(
            AcceptanceCriterion(
                id="review",
                description="Review the plan.",
                kind=CriterionKind.HUMAN_REVIEW,
            ),
        ),
    )

    context = ExperimentBuildAdapter().inspect(task)

    assert context.project_knowledge["project_id"] == "harness"
    assert context.project_knowledge["repo"] == str(tmp_path.resolve())
