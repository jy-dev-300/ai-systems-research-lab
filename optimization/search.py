from __future__ import annotations

from collections.abc import Callable, Iterable
from math import inf

from optimization.models import ExperimentResult, InferenceConfiguration

BenchmarkRunner = Callable[[InferenceConfiguration], ExperimentResult]


def search_configurations(
    configurations: Iterable[InferenceConfiguration],
    minimum_task_success: float,
    benchmark: BenchmarkRunner | None = None,
) -> list[ExperimentResult]:
    """Run a finite configuration grid and retain quality-acceptable results."""
    if not 0 <= minimum_task_success <= 1:
        raise ValueError("minimum_task_success must be between zero and one")
    accepted: list[ExperimentResult] = []
    for configuration in configurations:
        result = benchmark(configuration) if benchmark else _result_from_extra(configuration)
        if result.configuration != configuration:
            raise ValueError("benchmark result must describe the configuration that was run")
        if result.task_success_rate < minimum_task_success:
            continue
        accepted.append(result)
    return accepted


def pareto_frontier(results: Iterable[ExperimentResult]) -> list[ExperimentResult]:
    """Return nondominated quality and performance configurations."""
    candidates = list(results)
    frontier: list[ExperimentResult] = []
    for index, candidate in enumerate(candidates):
        if any(
            _dominates(other, candidate)
            for other_index, other in enumerate(candidates)
            if other_index != index
        ):
            continue
        frontier.append(candidate)
    return frontier


def _result_from_extra(configuration: InferenceConfiguration) -> ExperimentResult:
    required = (
        "task_success_rate",
        "p95_latency_ms",
        "throughput_requests_per_second",
    )
    missing = [name for name in required if name not in configuration.extra]
    if missing:
        raise ValueError(
            "provide a benchmark callback or measured configuration.extra fields: "
            + ", ".join(missing)
        )
    return ExperimentResult(
        configuration=configuration,
        task_success_rate=float(configuration.extra["task_success_rate"]),
        p95_latency_ms=float(configuration.extra["p95_latency_ms"]),
        throughput_requests_per_second=float(configuration.extra["throughput_requests_per_second"]),
        peak_gpu_memory_mb=_optional_float(configuration.extra.get("peak_gpu_memory_mb")),
        cost_per_task=_optional_float(configuration.extra.get("cost_per_task")),
        failures=int(configuration.extra.get("failures", 0)),
    )


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _dominates(left: ExperimentResult, right: ExperimentResult) -> bool:
    """Maximize success/throughput; minimize latency/memory/cost/failures.

    Missing minimization values are conservatively treated as positive infinity.
    Equal objective vectors do not dominate each other, so tied configurations remain.
    """
    left_values = (
        left.task_success_rate,
        left.throughput_requests_per_second,
        -left.p95_latency_ms,
        -(left.peak_gpu_memory_mb if left.peak_gpu_memory_mb is not None else inf),
        -(left.cost_per_task if left.cost_per_task is not None else inf),
        -float(left.failures),
    )
    right_values = (
        right.task_success_rate,
        right.throughput_requests_per_second,
        -right.p95_latency_ms,
        -(right.peak_gpu_memory_mb if right.peak_gpu_memory_mb is not None else inf),
        -(right.cost_per_task if right.cost_per_task is not None else inf),
        -float(right.failures),
    )
    no_worse = all(left >= right for left, right in zip(left_values, right_values, strict=True))
    strictly_better = any(
        left > right for left, right in zip(left_values, right_values, strict=True)
    )
    return no_worse and strictly_better
