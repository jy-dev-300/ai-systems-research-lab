from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence

from pydantic import BaseModel

from benchmarks.models import BenchmarkTask
from benchmarks.statistics import percentile
from harnesses.contracts import HarnessResult, RunStatus
from optimization.models import ExperimentResult, InferenceConfiguration
from profiling.models import RequestProfile, RequestStatus


class QuantizationExperimentReport(BaseModel):
    minimum_task_success: float
    results: list[ExperimentResult]
    eligible_configuration_names: list[str]


QuantizationRunner = Callable[
    [InferenceConfiguration, BenchmarkTask],
    Awaitable[tuple[HarnessResult, RequestProfile]],
]


async def run_quantization_experiment(
    configurations: Sequence[InferenceConfiguration],
    tasks: Sequence[BenchmarkTask],
    run_task: QuantizationRunner,
    *,
    minimum_task_success: float = 0.95,
) -> QuantizationExperimentReport:
    """Compare named artifacts/flags on identical tasks and enforce a quality floor."""
    if len(configurations) < 2:
        raise ValueError("compare at least two precision or quantization configurations")
    if not tasks:
        raise ValueError("at least one benchmark task is required")
    reports: list[ExperimentResult] = []
    for configuration in configurations:
        if not configuration.quantization:
            raise ValueError(f"{configuration.name} must name its precision or quantization")
        outcomes = [await run_task(configuration, task) for task in tasks]
        successes = sum(
            harness.status is RunStatus.SUCCEEDED
            and harness.final_answer == task.expected_answer
            and profile.status is RequestStatus.SUCCEEDED
            for (harness, profile), task in zip(outcomes, tasks, strict=True)
        )
        profiles = [profile for _, profile in outcomes]
        elapsed_seconds = sum(profile.total_latency_ms for profile in profiles) / 1_000
        memory = [p.memory_after_mb for p in profiles if p.memory_after_mb is not None]
        reports.append(
            ExperimentResult(
                configuration=configuration,
                task_success_rate=successes / len(tasks),
                p95_latency_ms=percentile((p.total_latency_ms for p in profiles), 0.95),
                throughput_requests_per_second=(
                    len(profiles) / elapsed_seconds if elapsed_seconds > 0 else 0.0
                ),
                peak_gpu_memory_mb=max(memory) if memory else None,
                cost_per_task=None,
                failures=len(tasks) - successes,
            )
        )
    return QuantizationExperimentReport(
        minimum_task_success=minimum_task_success,
        results=reports,
        eligible_configuration_names=[
            report.configuration.name
            for report in reports
            if report.task_success_rate >= minimum_task_success
        ],
    )
