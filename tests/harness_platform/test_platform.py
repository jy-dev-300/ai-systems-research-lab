from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from harness.adapters.base import AdapterEvaluation, AdapterRunOutput, TaskAdapter
from harness.adapters.registry import AdapterRegistry, adapter_registry, register_builtin_adapters
from harness.adapters.swebench_tools import RepositoryWorkspace
from harness.analysis.bottlenecks import analyze_bottlenecks
from harness.experiments.runner import ExperimentRunner
from harness.experiments.sweep import expand_sweep
from harness.failures import classify_failure
from harness.harness_registry import harness_registry
from harness.knowledge import ExperimentKnowledgeStore
from harness.models import (
    ExperimentConfig,
    ExperimentResult,
    FailureType,
    Metric,
)
from harness.research.advisor import ResearchAdvisor
from harness.research.ingestion import load_local_documents
from harness.tracking import ExperimentTracker
from harnesses.contracts import Harness


def config(tmp_path: Path, adapter: str = "fake") -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "config_version": "1.0",
            "experiment": {"name": "test", "domain": "test"},
            "adapter": {"name": adapter},
            "harness": {"type": "simple_loop"},
            "output_dir": str(tmp_path / "runs"),
        }
    )


class FakeAdapter(TaskAdapter):
    def load_tasks(self, config: ExperimentConfig) -> list[dict[str, Any]]:
        return [{"id": "ok"}]

    def setup(self, task: dict[str, Any], config: ExperimentConfig) -> None:
        pass

    async def run(
        self, task: dict[str, Any], harness: Harness, config: ExperimentConfig
    ) -> AdapterRunOutput:
        return AdapterRunOutput(value={"answer": 42}, artifact_payloads={"answer.json": {"n": 42}})

    def evaluate(
        self, task: dict[str, Any], run_output: AdapterRunOutput, config: ExperimentConfig
    ) -> AdapterEvaluation:
        return AdapterEvaluation(
            success=True,
            quality=1.0,
            metrics=[Metric(name="exact", value=1.0)],
            domain_metrics={"exact": 1.0},
        )


class BrokenAdapter(FakeAdapter):
    def setup(self, task: dict[str, Any], config: ExperimentConfig) -> None:
        raise FileNotFoundError("missing model")


class RepoCheckingAdapter(FakeAdapter):
    def setup(self, task: dict[str, Any], config: ExperimentConfig) -> None:
        assert self.assigned_repo == Path(str(config.adapter.parameters["repo"])).resolve()


class CapturingTracker(ExperimentTracker):
    def __init__(self) -> None:
        self.results: list[ExperimentResult] = []

    def log(self, result: ExperimentResult, run_dir: Path) -> None:
        self.results.append(result)


def test_config_parse_and_fingerprint(tmp_path: Path) -> None:
    parsed = config(tmp_path)
    assert parsed.config_version == "1.0"
    assert parsed.fingerprint() == parsed.fingerprint()


def test_result_serialization(tmp_path: Path) -> None:
    parsed = config(tmp_path)
    result = ExperimentResult(
        task_id="t",
        domain="test",
        model="m",
        provider="p",
        backend="b",
        harness_type="simple_loop",
        config_version="1",
        config_fingerprint="abc",
        config=parsed.model_dump(mode="json"),
        success=True,
    )
    assert ExperimentResult.model_validate_json(result.model_dump_json()).run_id == result.run_id


def test_adapter_and_harness_registries() -> None:
    register_builtin_adapters()
    assert {"bo", "swebench", "maniskill", "myvoice"}.issubset(adapter_registry.names())
    assert {"simple_loop", "planner_executor"}.issubset(harness_registry.names())


@pytest.mark.anyio
async def test_runner_success_and_tracking(tmp_path: Path) -> None:
    adapters = AdapterRegistry()
    adapters.register("fake", FakeAdapter)
    tracker = CapturingTracker()
    runner = ExperimentRunner(
        adapters=adapters,
        tracker=tracker,
        knowledge=ExperimentKnowledgeStore(tmp_path / "knowledge.sqlite3"),
    )
    parsed = config(tmp_path)
    parsed.parameters["lineage"] = {
        "build_task_id": "task-1",
        "build_result_id": "build-1",
        "research_hypothesis_id": "hypothesis-1",
        "experiment_plan_id": "plan-1",
    }
    result = await runner.run(parsed, {"id": "ok"})
    assert result.success
    assert result.build_task_id == "task-1"
    assert result.build_result_id == "build-1"
    assert tracker.results == [result]
    assert Path(result.artifacts[0].path).exists()


@pytest.mark.anyio
async def test_runner_failure_becomes_result(tmp_path: Path) -> None:
    adapters = AdapterRegistry()
    adapters.register("broken", BrokenAdapter)
    runner = ExperimentRunner(
        adapters=adapters,
        tracker=CapturingTracker(),
        knowledge=ExperimentKnowledgeStore(tmp_path / "knowledge.sqlite3"),
    )
    result = await runner.run(config(tmp_path, "broken"), {"id": "bad"})
    assert not result.success
    assert result.failure and result.failure.type is FailureType.RESOURCE


@pytest.mark.anyio
async def test_runner_binds_fresh_adapter_to_configured_repo(tmp_path: Path) -> None:
    adapters = AdapterRegistry()
    adapters.register("repo-checking", RepoCheckingAdapter)
    parsed = config(tmp_path, "repo-checking")
    parsed.adapter.parameters["repo"] = str(tmp_path)
    runner = ExperimentRunner(
        adapters=adapters,
        tracker=CapturingTracker(),
        knowledge=ExperimentKnowledgeStore(tmp_path / "knowledge.sqlite3"),
    )

    result = await runner.run(parsed, {"id": "bound"})

    assert result.success


def test_failure_classification() -> None:
    failure = classify_failure(
        error=None,
        evaluation=AdapterEvaluation(
            success=False,
            failure_subtype="relevant_chunk_not_retrieved",
            evidence=["missing c1"],
        ),
        run_output=AdapterRunOutput(),
    )
    assert failure and failure.type is FailureType.RETRIEVAL


def test_bottleneck_analysis(tmp_path: Path) -> None:
    parsed = config(tmp_path)
    result = ExperimentResult(
        task_id="t",
        domain="Bo",
        model="m",
        provider="p",
        backend="b",
        harness_type="simple_loop",
        config_version="1",
        config_fingerprint="abc",
        config=parsed.model_dump(mode="json"),
        success=False,
        domain_metrics={"recall_at_k": 0.2},
        model_calls=10,
    )
    kinds = {item.kind for item in analyze_bottlenecks(result).observations}
    assert {"low_retrieval_recall", "excessive_model_calls"}.issubset(kinds)


def test_research_candidate_schema(tmp_path: Path) -> None:
    parsed = config(tmp_path)
    result = ExperimentResult(
        task_id="t",
        domain="Bo",
        model="m",
        provider="p",
        backend="b",
        harness_type="simple_loop",
        config_version="1",
        config_fingerprint="abc",
        config=parsed.model_dump(mode="json"),
        success=False,
        domain_metrics={"recall_at_k": 0.0},
    )
    docs = load_local_documents(
        Path(__file__).parents[2] / "research/fixtures/research_documents.json"
    )
    candidate = ResearchAdvisor(docs).propose(result, analyze_bottlenecks(result))[0]
    assert candidate.internal_verification_status == "unverified"
    assert candidate.evidence_level == "external_evidence_only"


def test_parameter_sweep_generation(tmp_path: Path) -> None:
    base = config(tmp_path).model_dump(mode="json")
    configs = expand_sweep(
        {
            "base": base,
            "grid": {"harness.type": ["simple_loop", "planner_executor"], "seed": [1, 2]},
        }
    )
    assert len(configs) == 4
    assert {item.harness.type for item in configs} == {"simple_loop", "planner_executor"}


@pytest.mark.anyio
async def test_end_to_end_swebench_smoke(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    parsed = ExperimentConfig.from_file(root / "configs/swebench/simple_loop.yaml")
    parsed.output_dir = str(tmp_path / "runs")
    parsed.evaluation.dataset = str(root / "evals/swebench_smoke_v1.json")
    tracker = CapturingTracker()
    runner = ExperimentRunner(
        tracker=tracker,
        knowledge=ExperimentKnowledgeStore(tmp_path / "knowledge.sqlite3"),
    )
    result = (await runner.run_batch(parsed))[0]
    assert result.success
    assert result.domain_metrics["task_resolved"] is True
    assert {artifact.name for artifact in result.artifacts} >= {"final.patch", "test.log"}


def test_swebench_repository_tools_are_path_contained(tmp_path: Path) -> None:
    (tmp_path / "example.py").write_text("answer = 41\n", encoding="utf-8")
    workspace = RepositoryWorkspace(tmp_path)
    assert workspace.list_files("*.py") == ["example.py"]
    assert workspace.search("answer")[0]["line"] == 1
    workspace.write_file("example.py", "answer = 42\n")
    assert workspace.read_file("example.py") == "answer = 42\n"
    with pytest.raises(ValueError, match="escapes"):
        workspace.read_file("../outside.py")
