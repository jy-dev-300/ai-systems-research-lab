from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness.programs.classifier import JevWorkflowClassifier, RuleWorkflowClassifier
from harness.programs.dispatch import DispatchResult
from harness.programs.evaluation import evaluate_classifier
from harness.programs.models import (
    ProgramDocument,
    ProgramStatus,
    ProjectState,
    Workflow,
    WorkflowContext,
    WorkflowDecision,
)
from harness.programs.registry import ProgramRegistry
from harness.programs.runner import ProgramRunner


def context(text: str) -> WorkflowContext:
    program = ProgramDocument(
        path="bo-program.md",
        filename="bo-program.md",
        contents=text,
        sha256="0" * 64,
    )
    return WorkflowContext(
        program=program,
        project="Bo",
        adapter="bo",
        capabilities=("inspect", "build", "test", "evaluate", "experiment"),
        project_state=ProjectState(repo="../natural-language-file-search-rag", exists=True, summary="BM25 baseline"),
    )


def test_registry_resolves_default_and_multiple_program_names(tmp_path: Path) -> None:
    registry = ProgramRegistry(tmp_path)
    default = ProgramDocument.load(_write(tmp_path / "bo-program.md", "# Bo\nEvaluate it."))
    additional = ProgramDocument.load(
        _write(tmp_path / "bo-reranker-latency.md", "# Whatever\nMeasure latency.")
    )
    assert registry.resolve(default).adapter == "bo"
    assert registry.resolve(additional).project == "Bo"


def test_jev_classifier_uses_typed_choice_response() -> None:
    answer = SimpleNamespace(
        choice="RESEARCH",
        confidence=0.94,
        probabilities={"BUILD": 0.03, "EVAL": 0.03, "RESEARCH": 0.94},
    )

    class Client:
        def system_one(self, **kwargs):
            assert kwargs["questions"]["workflow"] is question
            assert kwargs["state"]["program"].startswith("Investigate")
            return SimpleNamespace(choices={"workflow": answer})

    question = object()
    decision = JevWorkflowClassifier(client=Client(), question=question).classify(
        context("Investigate whether fusion improves recall.")
    )
    assert decision.workflow is Workflow.RESEARCH
    assert decision.required_capabilities == (
        "inspect",
        "build",
        "test",
        "evaluate",
        "experiment",
    )
    assert decision.confidence == pytest.approx(0.94)


def test_rule_classifier_respects_top_level_research_and_eval() -> None:
    classifier = RuleWorkflowClassifier()
    assert (
        classifier.classify(context("Investigate whether adding RRF improves recall.")).workflow
        is Workflow.RESEARCH
    )
    assert (
        classifier.classify(context("Benchmark BM25. Do not modify it.")).workflow is Workflow.EVAL
    )


def test_routing_evaluation_reports_metrics(tmp_path: Path) -> None:
    dataset = {
        "version": "test-1",
        "cases": [
            {"id": "b", "program": "Implement the parser.", "gold": "BUILD"},
            {"id": "e", "program": "Measure current latency.", "gold": "EVAL"},
            {
                "id": "r",
                "program": "Investigate whether fusion improves recall.",
                "gold": "RESEARCH",
            },
        ],
    }
    path = tmp_path / "routing.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")
    report = evaluate_classifier(RuleWorkflowClassifier(), path)
    assert report.accuracy == 1.0
    assert set(report.per_class) == {"BUILD", "EVAL", "RESEARCH"}
    assert report.confusion_matrix["BUILD"]["BUILD"] == 1


def test_program_runner_preserves_exact_program_and_dispatches(tmp_path: Path) -> None:
    lab = tmp_path / "optimo"
    repo = tmp_path / "natural-language-file-search-rag"
    lab.mkdir()
    repo.mkdir()
    source = _write(lab / "bo-experiment.md", "# Bo\n\nMeasure current latency.\n")

    class Dispatcher:
        async def dispatch(self, **kwargs):
            assert kwargs["decision"].workflow is Workflow.EVAL
            assert kwargs["repo"] == repo
            assert kwargs["knowledge"].repo == str(repo)
            return DispatchResult(summary="measured", experiment_run_ids=("run-1",))

    result = asyncio.run(
        ProgramRunner(
            RuleWorkflowClassifier(),
            lab_root=lab,
            dispatcher=Dispatcher(),
            output_root=tmp_path / "artifacts",
        ).run(source)
    )
    assert result.status is ProgramStatus.COMPLETED
    assert result.experiment_run_ids == ("run-1",)
    assert result.project_knowledge["project_id"] == "bo"
    assert (Path(result.artifact_dir) / "adapter_knowledge.json").is_file()
    assert (Path(result.artifact_dir) / "program.md").read_text(
        encoding="utf-8"
    ) == source.read_text(encoding="utf-8")
    persisted = json.loads(
        (Path(result.artifact_dir) / "program_result.json").read_text(encoding="utf-8")
    )
    assert persisted["program"]["sha256"] == result.program.sha256


def test_low_confidence_ambiguous_program_requires_review(tmp_path: Path) -> None:
    lab = tmp_path / "optimo"
    (tmp_path / "natural-language-file-search-rag").mkdir()
    lab.mkdir()
    source = _write(lab / "bo-program.md", "# Bo\n\nLook at this.")

    class LowConfidence:
        def classify(self, _context):
            return WorkflowDecision(
                workflow=Workflow.EVAL,
                confidence=0.4,
                reason="uncertain",
                required_capabilities=("inspect", "evaluate"),
                classifier="Jev",
                classifier_version="test",
                model="fake",
                prompt_version="test",
            )

    result = asyncio.run(
        ProgramRunner(LowConfidence(), lab_root=lab, output_root=tmp_path / "ambiguous").run(source)
    )
    assert result.status is ProgramStatus.REVIEW_REQUIRED
    assert not result.experiment_run_ids


def test_parallel_program_runs_receive_distinct_state_directories(tmp_path: Path) -> None:
    lab = tmp_path / "optimo"
    (tmp_path / "natural-language-file-search-rag").mkdir()
    lab.mkdir()
    source = _write(lab / "bo-program.md", "# Bo\n\nMeasure latency.")

    class Dispatcher:
        async def dispatch(self, **_kwargs):
            return DispatchResult(summary="measured")

    runner = ProgramRunner(
        RuleWorkflowClassifier(),
        lab_root=lab,
        dispatcher=Dispatcher(),
        output_root=tmp_path / "parallel",
    )
    first = asyncio.run(runner.run(source))
    second = asyncio.run(runner.run(source))
    assert first.program_run_id != second.program_run_id
    assert first.artifact_dir != second.artifact_dir
    assert Path(first.artifact_dir).is_dir()
    assert Path(second.artifact_dir).is_dir()


def test_build_program_uses_isolated_worktree_before_dispatch(tmp_path: Path) -> None:
    lab = tmp_path / "optimo"
    repo = tmp_path / "natural-language-file-search-rag"
    isolated = tmp_path / "worktree-run"
    lab.mkdir()
    repo.mkdir()
    isolated.mkdir()
    source = _write(lab / "bo-program.md", "# Bo\n\nImplement a parser.")

    class Worktrees:
        def create(self, source_repo, run_id):
            assert Path(source_repo) == repo
            assert run_id
            return isolated, None, "abc123"

    class Dispatcher:
        async def dispatch(self, **kwargs):
            assert kwargs["repo"] == isolated
            assert kwargs["knowledge"].repo == str(isolated)
            return DispatchResult(summary="approval required", review_required=True)

    result = asyncio.run(
        ProgramRunner(
            RuleWorkflowClassifier(),
            lab_root=lab,
            dispatcher=Dispatcher(),
            worktrees=Worktrees(),
            output_root=tmp_path / "build-artifacts",
        ).run(source)
    )
    assert result.status is ProgramStatus.REVIEW_REQUIRED
    assert result.worktree == str(isolated)
    assert result.target_repo_commit == "abc123"


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path
