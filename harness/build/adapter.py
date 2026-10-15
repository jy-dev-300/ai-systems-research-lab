from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Protocol

from harness.adapters.registry import adapter_registry, register_builtin_adapters
from harness.build.models import (
    AcceptanceCriterion,
    AcceptanceCriterionResult,
    AppliedChange,
    BuildTask,
    BuildValidation,
    CriterionKind,
    RepositoryContext,
    TestExecutionRecord,
)
from harness.experiments.runner import ExperimentRunner
from harness.models import ExperimentConfig, ExperimentResult


class BuildTargetAdapter(Protocol):
    """Build-facing bridge that delegates domain evaluation to existing adapters."""

    def inspect(self, task: BuildTask) -> RepositoryContext: ...

    async def evaluate(
        self,
        task: BuildTask,
        changes: tuple[AppliedChange, ...],
        *,
        build_id: str,
    ) -> BuildValidation: ...

    def cleanup(self) -> None: ...


class AcceptanceEvaluator:
    """Evaluates the original BuildTask criteria without permitting redefinition."""

    OPERATORS = {
        ">=": lambda value, threshold: value >= threshold,
        ">": lambda value, threshold: value > threshold,
        "<=": lambda value, threshold: value <= threshold,
        "<": lambda value, threshold: value < threshold,
        "==": lambda value, threshold: value == threshold,
    }

    def evaluate(
        self,
        task: BuildTask,
        tests: tuple[TestExecutionRecord, ...],
        experiments: tuple[ExperimentResult, ...],
        changes: tuple[AppliedChange, ...],
    ) -> tuple[tuple[AcceptanceCriterionResult, ...], dict[str, float]]:
        metrics = self._metrics(experiments)
        results = tuple(
            self._criterion(task, criterion, tests, experiments, changes, metrics)
            for criterion in task.acceptance_criteria
        )
        return results, metrics

    def _criterion(
        self,
        task: BuildTask,
        criterion: AcceptanceCriterion,
        tests: tuple[TestExecutionRecord, ...],
        experiments: tuple[ExperimentResult, ...],
        changes: tuple[AppliedChange, ...],
        metrics: dict[str, float],
    ) -> AcceptanceCriterionResult:
        if criterion.kind is CriterionKind.TESTS_PASS:
            passed = bool(tests) and all(test.returncode == 0 for test in tests)
            evidence = tuple(f"{test.name}: exit {test.returncode}" for test in tests)
        elif criterion.kind is CriterionKind.EXPERIMENT_SUCCESS:
            passed = bool(experiments) and all(result.success for result in experiments)
            evidence = tuple(f"{result.run_id}: success={result.success}" for result in experiments)
        elif criterion.kind is CriterionKind.METRIC:
            name = criterion.metric_name or ""
            values = [
                result.metric_values()[name]
                for result in experiments
                if name in result.metric_values()
            ]
            if not values:
                value = None
            elif criterion.aggregation == "min":
                value = min(values)
            elif criterion.aggregation == "max":
                value = max(values)
            else:
                value = mean(values)
            passed = value is not None and self.OPERATORS[criterion.operator](
                value, float(criterion.threshold)
            )
            evidence = (f"{name}={value!r} {criterion.operator} {criterion.threshold}",)
        elif criterion.kind is CriterionKind.FILE_EXISTS:
            root = Path(task.target_project).resolve()
            path = (root / str(criterion.path)).resolve()
            contained = path == root or root in path.parents
            passed = contained and path.is_file()
            evidence = (f"{criterion.path}: exists={passed}",)
        elif criterion.kind is CriterionKind.DIFF_NONEMPTY:
            passed = bool(changes)
            evidence = (f"applied_changes={len(changes)}",)
        else:
            passed = False
            evidence = ("human review is required for this criterion",)
        return AcceptanceCriterionResult(
            criterion_id=criterion.id,
            passed=passed,
            evidence=evidence,
        )

    @staticmethod
    def _metrics(experiments: tuple[ExperimentResult, ...]) -> dict[str, float]:
        grouped: dict[str, list[float]] = {}
        for result in experiments:
            for name, value in result.metric_values().items():
                grouped.setdefault(name, []).append(value)
        return {name: mean(values) for name, values in grouped.items()}


class ExperimentBuildAdapter:
    def __init__(
        self,
        runner: ExperimentRunner | None = None,
        evaluator: AcceptanceEvaluator | None = None,
    ) -> None:
        self.runner = runner or ExperimentRunner()
        self.evaluator = evaluator or AcceptanceEvaluator()

    def inspect(self, task: BuildTask) -> RepositoryContext:
        root = Path(task.target_project).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Target project not found: {root}")
        files = tuple(
            sorted(
                path.relative_to(root).as_posix()
                for path in root.rglob("*")
                if path.is_file()
                and not any(
                    part in {".git", ".venv", "node_modules", "__pycache__"} for part in path.parts
                )
            )[:2_000]
        )
        contents: dict[str, str] = {}
        register_builtin_adapters()
        domain_adapter = adapter_registry.create(task.adapter_name, repo=root)
        knowledge = domain_adapter.project_knowledge()
        requested_files = tuple(dict.fromkeys((*task.relevant_files, *knowledge.entry_points)))
        for relative in requested_files[:20]:
            path = (root / relative).resolve()
            if root not in path.parents and path != root:
                raise ValueError(f"Relevant file escapes target project: {relative}")
            if path.is_file():
                contents[relative] = path.read_text(encoding="utf-8")[:20_000]
        commit, dirty = self._git_state(root)
        return RepositoryContext(
            root=str(root),
            files=files,
            relevant_contents=contents,
            git_commit=commit,
            dirty=dirty,
            project_knowledge=knowledge.model_dump(mode="json"),
        )

    async def evaluate(
        self,
        task: BuildTask,
        changes: tuple[AppliedChange, ...],
        *,
        build_id: str,
    ) -> BuildValidation:
        tests = tuple(self._run_command(task, command) for command in task.validation_commands)
        experiment_results: list[ExperimentResult] = []
        for config_path in task.validation_config_paths:
            config = ExperimentConfig.from_file(config_path).model_copy(deep=True)
            config.adapter.parameters["repo"] = str(Path(task.target_project).resolve())
            lineage = config.parameters.setdefault("lineage", {})
            lineage.update(
                {
                    "build_task_id": task.id,
                    "build_result_id": build_id,
                    "research_hypothesis_id": task.research_hypothesis_id,
                    "experiment_plan_id": task.experiment_plan_id,
                }
            )
            experiment_results.extend(await self.runner.run_batch(config))
        experiments = tuple(experiment_results)
        criterion_results, metrics = self.evaluator.evaluate(task, tests, experiments, changes)
        failures = tuple(
            [f"test {test.name} exited {test.returncode}" for test in tests if test.returncode]
            + [f"experiment {result.run_id} failed" for result in experiments if not result.success]
            + [
                f"criterion {criterion.criterion_id} failed"
                for criterion in criterion_results
                if not criterion.passed
            ]
        )
        return BuildValidation(
            criterion_results=criterion_results,
            tests=tests,
            experiment_results=experiments,
            metrics=metrics,
            failures=failures,
            requires_human_review=any(
                criterion.kind is CriterionKind.HUMAN_REVIEW
                for criterion in task.acceptance_criteria
            ),
        )

    def _run_command(self, task: BuildTask, command: object) -> TestExecutionRecord:
        from harness.build.models import ValidationCommand

        assert isinstance(command, ValidationCommand)
        started = perf_counter()
        try:
            process = subprocess.run(
                list(command.command),
                cwd=Path(task.target_project).resolve(),
                capture_output=True,
                text=True,
                timeout=command.timeout_seconds,
                check=False,
            )
            return TestExecutionRecord(
                name=command.name,
                command=command.command,
                returncode=process.returncode,
                stdout=process.stdout[-20_000:],
                stderr=process.stderr[-20_000:],
                duration_ms=(perf_counter() - started) * 1_000,
            )
        except subprocess.TimeoutExpired as error:
            return TestExecutionRecord(
                name=command.name,
                command=command.command,
                returncode=124,
                stdout=str(error.stdout or "")[-20_000:],
                stderr=f"Timed out after {command.timeout_seconds}s",
                duration_ms=(perf_counter() - started) * 1_000,
            )

    @staticmethod
    def _git_state(root: Path) -> tuple[str | None, bool | None]:
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if commit.returncode:
                return None, None
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            return commit.stdout.strip(), bool(status.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            return None, None

    def cleanup(self) -> None:
        return None


BuildAdapterFactory = Callable[[], BuildTargetAdapter]


class BuildAdapterRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, BuildAdapterFactory] = {}

    def register(self, name: str, factory: BuildAdapterFactory, *, replace: bool = False) -> None:
        if name in self._factories and not replace:
            raise ValueError(f"Build adapter already registered: {name}")
        self._factories[name] = factory

    def create(self, name: str) -> BuildTargetAdapter:
        try:
            return self._factories[name]()
        except KeyError as error:
            choices = ", ".join(sorted(self._factories)) or "none"
            raise ValueError(f"Unknown build adapter {name!r}; available: {choices}") from error


build_adapter_registry = BuildAdapterRegistry()
for _name in ("bo", "myvoice", "swebench", "maniskill", "harness"):
    build_adapter_registry.register(_name, ExperimentBuildAdapter)
