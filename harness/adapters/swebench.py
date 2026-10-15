from __future__ import annotations

from pathlib import Path
from typing import Any

from harness.adapters.base import AdapterEvaluation, AdapterRunOutput, TaskAdapter
from harness.adapters.common import load_json_records, resolve_workspace_path, run_reasoning_harness
from harness.adapters.knowledge import (
    AdapterCapability,
    EvaluationKnowledge,
    MetricDirection,
    MetricKnowledge,
    ProjectKnowledge,
)
from harness.adapters.swebench_tools import RepositoryWorkspace
from harness.models import ExperimentConfig, Metric
from harnesses.contracts import Harness


class SWEbenchAdapter(TaskAdapter):
    """SWE-bench boundary with a runnable local patch-evaluation smoke path."""

    def __init__(self) -> None:
        self.workspace: RepositoryWorkspace | None = None

    def project_knowledge(self) -> ProjectKnowledge:
        repo = self.assigned_repo or Path("../swe-bench").resolve()
        return ProjectKnowledge(
            project_id="swebench",
            project_name="SWE-bench",
            project_type="coding_agent_benchmark",
            repo=str(repo),
            summary="Local patch smoke tasks plus an official SWE-bench repository/test boundary.",
            capabilities=frozenset(AdapterCapability),
            editable_paths=("task-assigned repository worktree",),
            protected_paths=(
                "evals/swebench_smoke_v1.json",
                "official task metadata and gold patches",
            ),
            entry_points=("harness/adapters/swebench_tools.py",),
            evaluation=EvaluationKnowledge(
                operation=(
                    "Run task-specific tests and official SWE-bench evaluation when installed"
                ),
                dataset="evals/swebench_smoke_v1.json",
                dataset_version="swebench-smoke-v1",
                split="configurable task subset",
                metrics=(
                    MetricKnowledge(
                        name="task_resolved", direction=MetricDirection.HIGHER, primary=True
                    ),
                    MetricKnowledge(name="tests_passed", direction=MetricDirection.HIGHER),
                    MetricKnowledge(name="tests_failed", direction=MetricDirection.LOWER),
                    MetricKnowledge(name="model_calls", direction=MetricDirection.LOWER),
                    MetricKnowledge(name="tool_calls", direction=MetricDirection.LOWER),
                    MetricKnowledge(name="tokens", direction=MetricDirection.LOWER),
                    MetricKnowledge(name="cost_usd", direction=MetricDirection.LOWER),
                ),
            ),
            baseline="configs/swebench/simple_loop.yaml on the same immutable task subset.",
            configurable_components=("harness type", "task subset", "max steps", "retry policy"),
            reproducibility_requirements=(
                "Use the same task repository commit, issue, tests, and container image.",
            ),
            artifacts_to_preserve=("final.patch", "test.log", "agent output", "trace"),
            integrity_constraints=(
                "Do not modify tests, task metadata, gold patches, or official evaluator logic.",
            ),
            environment_assumptions=(
                "Official mode requires the SWE-bench checkout, datasets, task repositories, "
                "and containers.",
            ),
        )

    def load_tasks(self, config: ExperimentConfig) -> list[dict[str, Any]]:
        dataset = config.evaluation.dataset or "evals/swebench_smoke_v1.json"
        tasks = load_json_records(resolve_workspace_path(config, dataset))
        limit = int(config.adapter.parameters.get("limit", len(tasks)))
        return tasks[:limit]

    def setup(self, task: dict[str, Any], config: ExperimentConfig) -> None:
        mode = config.adapter.parameters.get("mode", "smoke")
        repo = self.assigned_repo or resolve_workspace_path(
            config, str(config.adapter.parameters.get("repo", "../swe-bench"))
        )
        if mode == "official" and not any(repo.iterdir()):
            raise RuntimeError(
                "SWE-bench checkout is empty; install SWE-bench and its dataset "
                "before official mode"
            )
        task_repo = task.get("repo_path")
        if mode == "official" and task_repo:
            self.workspace = RepositoryWorkspace(resolve_workspace_path(config, str(task_repo)))

    async def run(
        self, task: dict[str, Any], harness: Harness, config: ExperimentConfig
    ) -> AdapterRunOutput:
        if config.adapter.parameters.get("mode", "smoke") == "official":
            return await self._run_official(task, harness, config)
        # Smoke records are representative, immutable patch/test outcomes. The
        # fixture uses the same fields produced by the official path.
        patch = str(task.get("candidate_patch", ""))
        test_log = str(task.get("test_log", "smoke test: passed"))
        resolved = bool(task.get("resolved", bool(patch)))
        summary = "resolved" if resolved else "unresolved"
        harness_result = await run_reasoning_harness(
            harness,
            task_id=str(task["instance_id"]),
            prompt=str(task["problem_statement"]),
            expected_answer=summary,
        )
        return AdapterRunOutput(
            value={"patch": patch, "test_log": test_log, "resolved": resolved},
            harness_result=harness_result,
            metrics={
                "tests_passed": int(task.get("tests_passed", resolved)),
                "tests_failed": int(task.get("tests_failed", not resolved)),
                "files_modified": int(task.get("files_modified", 1 if patch else 0)),
                "patch_size": len(patch.splitlines()),
            },
            artifact_payloads={
                "final.patch": patch,
                "test.log": test_log,
                "agent_output.txt": harness_result.final_answer or "",
            },
        )

    async def _run_official(
        self, task: dict[str, Any], harness: Harness, config: ExperimentConfig
    ) -> AdapterRunOutput:
        if self.workspace is None:
            raise ValueError("Official SWE-bench tasks require repo_path")
        edits = task.get("candidate_edits", {})
        if not isinstance(edits, dict):
            raise ValueError("candidate_edits must map relative paths to file contents")
        for relative, content in edits.items():
            self.workspace.write_file(str(relative), str(content))
        test_command = task.get("test_command")
        if not isinstance(test_command, list) or not all(
            isinstance(part, str) for part in test_command
        ):
            raise ValueError("Official SWE-bench tasks require test_command as a string list")
        tests = self.workspace.run_tests(
            test_command,
            timeout_seconds=float(config.adapter.parameters.get("test_timeout_seconds", 300)),
        )
        diff = self.workspace.run_tests(["git", "diff", "--no-ext-diff"])
        patch = diff.stdout
        resolved = tests.returncode == 0
        harness_result = await run_reasoning_harness(
            harness,
            task_id=str(task["instance_id"]),
            prompt=str(task["problem_statement"]),
            expected_answer="resolved" if resolved else "unresolved",
        )
        return AdapterRunOutput(
            value={
                "patch": patch,
                "test_log": tests.stdout + tests.stderr,
                "resolved": resolved,
            },
            harness_result=harness_result,
            metrics={
                "tests_passed": int(resolved),
                "tests_failed": int(not resolved),
                "files_modified": len(edits),
                "patch_size": len(patch.splitlines()),
            },
            artifact_payloads={"final.patch": patch, "test.log": tests.stdout + tests.stderr},
        )

    def evaluate(
        self, task: dict[str, Any], run_output: AdapterRunOutput, config: ExperimentConfig
    ) -> AdapterEvaluation:
        resolved = bool(run_output.value["resolved"])
        metrics = dict(run_output.metrics)
        return AdapterEvaluation(
            success=resolved,
            quality=float(resolved),
            metrics=[Metric(name=name, value=float(value)) for name, value in metrics.items()],
            domain_metrics={"task_resolved": resolved, **metrics},
            failure_subtype=None if resolved else "tests_still_failing",
            evidence=[] if resolved else [str(run_output.value["test_log"])],
            details={"mode": config.adapter.parameters.get("mode", "smoke")},
            dataset=config.evaluation.dataset,
            dataset_version="swebench-smoke-v1"
            if config.adapter.parameters.get("mode", "smoke") == "smoke"
            else None,
        )

    def cleanup(self) -> None:
        self.workspace = None
