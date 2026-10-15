from __future__ import annotations

from pathlib import Path
from typing import Any

from harness.adapters.base import AdapterEvaluation, AdapterRunOutput, TaskAdapter
from harness.adapters.common import load_json_records, resolve_workspace_path, run_reasoning_harness
from harness.adapters.knowledge import (
    AdapterCapability,
    CommandKnowledge,
    EvaluationKnowledge,
    MetricDirection,
    MetricKnowledge,
    ProjectKnowledge,
)
from harness.models import ExperimentConfig, Metric
from harnesses.contracts import Harness, RunStatus


class HarnessAdapter(TaskAdapter):
    """Makes the Lab's execution harnesses measurable through their own experiment engine."""

    def load_tasks(self, config: ExperimentConfig) -> list[dict[str, Any]]:
        dataset = config.evaluation.dataset or "evals/harness_smoke_v1.json"
        return load_json_records(resolve_workspace_path(config, dataset))

    def setup(self, task: dict[str, Any], config: ExperimentConfig) -> None:
        if "expected_answer" not in task:
            raise ValueError("Harness task requires expected_answer")

    async def run(
        self, task: dict[str, Any], harness: Harness, config: ExperimentConfig
    ) -> AdapterRunOutput:
        result = await run_reasoning_harness(
            harness,
            task_id=str(task["id"]),
            prompt=str(task["prompt"]),
            expected_answer=str(task["expected_answer"]),
        )
        return AdapterRunOutput(
            value={"answer": result.final_answer or ""},
            harness_result=result,
            artifact_payloads={"agent_output.txt": result.final_answer or ""},
        )

    def evaluate(
        self, task: dict[str, Any], run_output: AdapterRunOutput, config: ExperimentConfig
    ) -> AdapterEvaluation:
        expected = str(task["expected_answer"]).strip().casefold()
        actual = str(run_output.value["answer"]).strip().casefold()
        succeeded = actual == expected and (
            run_output.harness_result is not None
            and run_output.harness_result.status is RunStatus.SUCCEEDED
        )
        quality = 1.0 if succeeded else 0.0
        return AdapterEvaluation(
            success=succeeded,
            quality=quality,
            metrics=[Metric(name="task_success", value=quality)],
            domain_metrics={"task_success": succeeded},
            failure_subtype=None if succeeded else "harness_answer_mismatch",
            evidence=[] if succeeded else [f"expected={expected!r}, actual={actual!r}"],
            dataset=config.evaluation.dataset,
            dataset_version="harness-smoke-v1",
        )

    def project_knowledge(self) -> ProjectKnowledge:
        repo = self.assigned_repo or Path.cwd().resolve()
        return ProjectKnowledge(
            project_id="harness",
            project_name="Optimo Harness",
            project_type="agent_harness",
            repo=str(repo),
            summary=(
                "Simple loop, planner/executor, retries, tracing, profiling, and A-M experiments."
            ),
            capabilities=frozenset(AdapterCapability),
            editable_paths=(
                "harness/",
                "harnesses/",
                "tools/",
                "tracing/",
                "profiling/",
                "optimization/",
                "tests/",
            ),
            protected_paths=("evals/", "tests/hardware/"),
            entry_points=(
                "harnesses/simple_tool_loop.py",
                "harnesses/planner_executor.py",
                "harness/experiments/runner.py",
                "harness/harness_registry.py",
            ),
            test_commands=(
                CommandKnowledge(
                    name="lab-tests",
                    command=("python", "-m", "pytest", "-q"),
                    purpose="Run harness, platform, and integration tests.",
                    timeout_seconds=900,
                ),
                CommandKnowledge(
                    name="lint",
                    command=("python", "-m", "ruff", "check", "."),
                    purpose="Run static checks.",
                ),
            ),
            evaluation=EvaluationKnowledge(
                operation="ExperimentRunner with HarnessAdapter on fixed harness tasks",
                dataset="evals/harness_smoke_v1.json",
                dataset_version="harness-smoke-v1",
                split="fixed harness smoke tasks",
                metrics=(
                    MetricKnowledge(
                        name="task_success", direction=MetricDirection.HIGHER, primary=True
                    ),
                    MetricKnowledge(name="model_calls", direction=MetricDirection.LOWER),
                    MetricKnowledge(name="tool_calls", direction=MetricDirection.LOWER),
                    MetricKnowledge(name="retries", direction=MetricDirection.LOWER),
                    MetricKnowledge(name="tokens", direction=MetricDirection.LOWER),
                    MetricKnowledge(name="latency_ms", direction=MetricDirection.LOWER, unit="ms"),
                    MetricKnowledge(name="cost_usd", direction=MetricDirection.LOWER, unit="USD"),
                ),
            ),
            baseline="configs/harness/baseline.yaml using simple_loop and deterministic backend.",
            configurable_components=(
                "harness type",
                "max steps",
                "retry policy",
                "tool policy",
                "prefix/cache strategy",
            ),
            reproducibility_requirements=(
                "Hold tasks, model, backend, seed, and limits constant.",
            ),
            artifacts_to_preserve=("trace", "config", "evaluation", "exact patch"),
            integrity_constraints=(
                "Do not modify held-out tasks or success criteria after observing results.",
                "Mock results prove control flow only, not model or hardware performance.",
            ),
            environment_assumptions=(
                "Hardware paths require their declared servers and accelerators.",
            ),
        )
