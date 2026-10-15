from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable, Iterable, Sequence
from time import perf_counter

from pydantic import BaseModel

from benchmarks.models import BenchmarkTask
from benchmarks.statistics import mean_optional, percentile
from optimization.models import InferenceConfiguration
from profiling.models import RequestProfile, RequestStatus


class ConcurrencyResult(BaseModel):
    configuration: InferenceConfiguration
    requests: int
    throughput_requests_per_second: float
    p95_latency_ms: float
    mean_queue_delay_ms: float | None
    failures: int
    oom_failures: int


ConcurrencyRunner = Callable[
    [InferenceConfiguration, BenchmarkTask, str], Awaitable[RequestProfile]
]


async def run_concurrency_experiment(
    configurations: Iterable[InferenceConfiguration],
    tasks: Sequence[BenchmarkTask],
    run_request: ConcurrencyRunner,
    *,
    repetitions: int = 3,
    poisson_rate_per_second: float = 20.0,
    seed: int = 11,
) -> list[ConcurrencyResult]:
    """Run the same task distribution under each concurrency/batching condition."""
    if not tasks or repetitions < 1 or poisson_rate_per_second <= 0:
        raise ValueError("tasks, repetitions, and Poisson rate must be positive")
    workload = [task for _ in range(repetitions) for task in tasks]
    reports: list[ConcurrencyResult] = []

    for configuration in configurations:
        semaphore = asyncio.Semaphore(configuration.concurrency)
        rng = random.Random(seed)

        async def execute(
            index: int,
            task: BenchmarkTask,
            active_configuration: InferenceConfiguration = configuration,
            active_semaphore: asyncio.Semaphore = semaphore,
        ) -> RequestProfile:
            async with active_semaphore:
                request_id = f"{active_configuration.name}-{index}"
                request_started = perf_counter()
                try:
                    return await run_request(active_configuration, task, request_id)
                except Exception as error:
                    return RequestProfile(
                        request_id=request_id,
                        status=RequestStatus.FAILED,
                        time_to_first_token_ms=None,
                        total_latency_ms=(perf_counter() - request_started) * 1_000,
                        queue_time_ms=None,
                        input_tokens=0,
                        output_tokens=0,
                        output_tokens_per_second=None,
                        memory_before_mb=None,
                        memory_after_mb=None,
                        error=f"{type(error).__name__}: {error}",
                    )

        started = perf_counter()
        if configuration.arrival_pattern == "serial":
            profiles = [await execute(index, task) for index, task in enumerate(workload)]
        else:
            pending = []
            for index, task in enumerate(workload):
                pending.append(asyncio.create_task(execute(index, task)))
                if configuration.arrival_pattern == "poisson" and index < len(workload) - 1:
                    await asyncio.sleep(rng.expovariate(poisson_rate_per_second))
            profiles = list(await asyncio.gather(*pending))
        elapsed = max(perf_counter() - started, 1e-9)
        successful = [p for p in profiles if p.status is RequestStatus.SUCCEEDED]
        failures = [p for p in profiles if p.status is not RequestStatus.SUCCEEDED]
        reports.append(
            ConcurrencyResult(
                configuration=configuration,
                requests=len(profiles),
                throughput_requests_per_second=len(successful) / elapsed,
                p95_latency_ms=percentile((p.total_latency_ms for p in profiles), 0.95),
                mean_queue_delay_ms=mean_optional(p.queue_time_ms for p in profiles),
                failures=len(failures),
                oom_failures=sum("oom" in (p.error or "").lower() for p in failures),
            )
        )
    return reports
