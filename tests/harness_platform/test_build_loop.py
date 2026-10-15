from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from harness.build.adapter import (
    BuildAdapterRegistry,
    BuildTargetAdapter,
    ExperimentBuildAdapter,
)
from harness.build.agent import BuildAgent
from harness.build.backend import ScriptedBuildBackend
from harness.build.loop import BuildLoop, review_build
from harness.build.models import (
    AcceptanceCriterion,
    AcceptanceCriterionResult,
    BuildApprovalState,
    BuildProposal,
    BuildResult,
    BuildStatus,
    BuildTask,
    BuildValidation,
    CriterionKind,
    FileChange,
    RepositoryContext,
    ValidationCommand,
)
from harness.build.research import build_task_from_research, hypothesis_from_candidate
from harness.research.models import CandidateExperiment, ExperimentPlan


def build_task(tmp_path: Path, **updates: object) -> BuildTask:
    values: dict[str, object] = {
        "target_project": str(tmp_path),
        "adapter_name": "fake",
        "goal": "Make the fixture pass",
        "acceptance_criteria": (
            AcceptanceCriterion(
                id="tests",
                description="Fixture validation passes",
                kind=CriterionKind.TESTS_PASS,
            ),
        ),
        "approval_state": BuildApprovalState.APPROVED,
        "output_dir": str(tmp_path / "builds"),
    }
    values.update(updates)
    return BuildTask.model_validate(values)


class FileCheckingAdapter(BuildTargetAdapter):
    def __init__(self, target: Path, expected: str = "good") -> None:
        self.target = target
        self.expected = expected
        self.evaluations = 0
        self.cleaned = False

    def inspect(self, task: BuildTask) -> RepositoryContext:
        return RepositoryContext(root=str(self.target), files=tuple())

    async def evaluate(self, task: BuildTask, changes: tuple, *, build_id: str) -> BuildValidation:
        self.evaluations += 1
        passed = (self.target / "feature.txt").read_text(encoding="utf-8") == self.expected
        return BuildValidation(
            criterion_results=(
                AcceptanceCriterionResult(
                    criterion_id="tests", passed=passed, evidence=(f"content={passed}",)
                ),
            ),
            failures=() if passed else ("fixture validation failed",),
        )

    def cleanup(self) -> None:
        self.cleaned = True


def registry(adapter: BuildTargetAdapter) -> BuildAdapterRegistry:
    result = BuildAdapterRegistry()
    result.register("fake", lambda: adapter)
    return result


def proposal(content: str, *, operation: str = "create", **updates: object) -> BuildProposal:
    values: dict[str, object] = {
        "summary": f"Write {content}",
        "changes": (FileChange(path="feature.txt", operation=operation, content=content),),
    }
    values.update(updates)
    return BuildProposal.model_validate(values)


def test_build_task_is_typed_and_criteria_are_fixed(tmp_path: Path) -> None:
    task = build_task(tmp_path)
    assert task.spec_fingerprint() == task.spec_fingerprint()
    with pytest.raises(ValidationError):
        task.acceptance_criteria[0].description = "changed"
    with pytest.raises(ValidationError, match="unique"):
        build_task(
            tmp_path,
            acceptance_criteria=(
                AcceptanceCriterion(id="same", description="a", kind="tests_pass"),
                AcceptanceCriterion(id="same", description="b", kind="tests_pass"),
            ),
        )


@pytest.mark.anyio
async def test_unapproved_task_does_not_modify_target(tmp_path: Path) -> None:
    task = build_task(tmp_path, approval_state=BuildApprovalState.PROPOSED)
    adapter = FileCheckingAdapter(tmp_path)
    result = await BuildLoop(
        BuildAgent(ScriptedBuildBackend([proposal("good")])), adapters=registry(adapter)
    ).run(task)
    assert result.final_status is BuildStatus.HUMAN_REVIEW_REQUIRED
    assert result.approval_state is BuildApprovalState.PROPOSED
    assert not (tmp_path / "feature.txt").exists()
    assert adapter.evaluations == 0


@pytest.mark.anyio
async def test_build_loop_success_requires_adapter_evidence(tmp_path: Path) -> None:
    adapter = FileCheckingAdapter(tmp_path)
    result = await BuildLoop(
        BuildAgent(ScriptedBuildBackend([proposal("good")])), adapters=registry(adapter)
    ).run(build_task(tmp_path))
    assert result.final_status is BuildStatus.SUCCESS
    assert result.approval_state is BuildApprovalState.REVIEW_REQUIRED
    assert result.acceptance_criteria_results[0].passed
    assert result.files_changed == ("feature.txt",)
    assert result.diff_artifact and Path(result.diff_artifact).exists()
    assert adapter.evaluations == 1 and adapter.cleaned
    reviewed = review_build(result, accept=True)
    assert reviewed.approval_state is BuildApprovalState.ACCEPTED


@pytest.mark.anyio
async def test_build_loop_diagnoses_and_revises(tmp_path: Path) -> None:
    adapter = FileCheckingAdapter(tmp_path)
    backend = ScriptedBuildBackend([proposal("bad"), proposal("good", operation="update")])
    result = await BuildLoop(BuildAgent(backend), adapters=registry(adapter)).run(
        build_task(tmp_path, max_iterations=2)
    )
    assert result.final_status is BuildStatus.SUCCESS
    assert len(result.iterations) == 2
    assert result.iterations[0].diagnosis is not None
    assert result.failures_encountered == ("fixture validation failed",)


@pytest.mark.anyio
async def test_iteration_limit_is_explicit(tmp_path: Path) -> None:
    adapter = FileCheckingAdapter(tmp_path, expected="never")
    backend = ScriptedBuildBackend([proposal("bad"), proposal("still-bad", operation="update")])
    result = await BuildLoop(BuildAgent(backend), adapters=registry(adapter)).run(
        build_task(tmp_path, max_iterations=1)
    )
    assert result.final_status is BuildStatus.LIMIT_REACHED
    assert "iteration limit reached" in result.failures_encountered[-1]


@pytest.mark.anyio
async def test_token_limit_stops_before_changes(tmp_path: Path) -> None:
    adapter = FileCheckingAdapter(tmp_path)
    expensive = proposal("good", tokens_used=11)
    result = await BuildLoop(
        BuildAgent(ScriptedBuildBackend([expensive])), adapters=registry(adapter)
    ).run(build_task(tmp_path, max_tokens=10))
    assert result.final_status is BuildStatus.LIMIT_REACHED
    assert not (tmp_path / "feature.txt").exists()
    assert adapter.evaluations == 0


@pytest.mark.anyio
async def test_disallowed_or_destructive_action_requires_review(tmp_path: Path) -> None:
    adapter = FileCheckingAdapter(tmp_path)
    architecture = proposal("good", actions=frozenset({"architecture"}))
    result = await BuildLoop(
        BuildAgent(ScriptedBuildBackend([architecture])), adapters=registry(adapter)
    ).run(build_task(tmp_path))
    assert result.final_status is BuildStatus.HUMAN_REVIEW_REQUIRED
    assert not (tmp_path / "feature.txt").exists()


@pytest.mark.anyio
async def test_experiment_build_adapter_executes_approved_validation_command(
    tmp_path: Path,
) -> None:
    task = build_task(
        tmp_path,
        adapter_name="unused",
        validation_commands=(
            ValidationCommand(
                name="python_smoke",
                command=("py", "-c", "print('ok')"),
                timeout_seconds=10,
            ),
        ),
    )
    validation = await ExperimentBuildAdapter().evaluate(task, tuple(), build_id="build-1")
    assert validation.accepted
    assert validation.tests[0].returncode == 0


def test_build_result_serialization(tmp_path: Path) -> None:
    task = build_task(tmp_path)
    result = BuildResult(
        task=task,
        spec_fingerprint=task.spec_fingerprint(),
        final_status=BuildStatus.FAILED,
        approval_state=BuildApprovalState.REVIEW_REQUIRED,
    )
    restored = BuildResult.model_validate_json(result.model_dump_json())
    assert restored.build_id == result.build_id


def test_research_candidate_creates_proposed_linked_build_task(tmp_path: Path) -> None:
    candidate = CandidateExperiment(
        observed_bottleneck="low_retrieval_recall",
        candidate_technique="hybrid retrieval",
        mechanism="combine complementary rankings",
        expected_metric_affected=["recall_at_k"],
        controlled_experiment="Compare BM25 with BM25 plus dense RRF.",
        evidence_level="external_evidence_only",
        confidence=0.6,
    )
    hypothesis = hypothesis_from_candidate(candidate)
    plan = ExperimentPlan(
        hypothesis_id=hypothesis.id,
        control_config="bm25.yaml",
        treatment_config="hybrid.yaml",
        verification_rules=["recall does not regress"],
    )
    task = build_task_from_research(
        candidate=candidate,
        hypothesis=hypothesis,
        experiment_plan=plan,
        target_project=str(tmp_path),
        adapter_name="bo",
        requirements=("add hybrid retrieval",),
        constraints=("preserve baseline",),
        acceptance_criteria=(
            AcceptanceCriterion(id="eval", description="eval passes", kind="experiment_success"),
        ),
        validation_config_paths=("hybrid.yaml",),
    )
    assert task.approval_state is BuildApprovalState.PROPOSED
    assert task.research_hypothesis_id == hypothesis.id
    assert task.experiment_plan_id == plan.id
