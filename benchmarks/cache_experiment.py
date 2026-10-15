from __future__ import annotations

import random
from collections.abc import Awaitable, Callable
from enum import StrEnum

from pydantic import BaseModel

from benchmarks.statistics import mean_optional, percentile
from profiling.models import RequestProfile, RequestStatus


class CacheCondition(StrEnum):
    DISABLED = "disabled"
    ENABLED_UNIQUE_PREFIX = "enabled_unique_prefix"
    ENABLED_REPEATED_PREFIX = "enabled_repeated_prefix"


class CacheRequest(BaseModel):
    condition: CacheCondition
    prefix_cache_enabled: bool
    prefix: str
    sample_index: int
    warmup: bool = False


class CacheConditionResult(BaseModel):
    condition: CacheCondition
    samples: int
    failures: int
    p50_latency_ms: float
    p95_latency_ms: float
    mean_output_tokens_per_second: float | None
    peak_memory_mb: float | None
    cache_hit_rate: float | None


class CacheExperimentReport(BaseModel):
    warmups_per_condition: int
    repetitions: int
    randomized_seed: int
    conditions: list[CacheConditionResult]
    interpretation: str


CacheRunner = Callable[[CacheRequest], Awaitable[RequestProfile]]


async def run_cache_experiment(
    run_request: CacheRunner,
    *,
    repetitions: int = 10,
    warmups_per_condition: int = 2,
    seed: int = 7,
) -> CacheExperimentReport:
    """Compare disabled, unique-prefix, and byte-identical-prefix conditions.

    Prefill processes input tokens; decode generates new tokens. Prefix caching
    can avoid repeated prefill work, but lookup/bookkeeping may cost more than it
    saves for short or mostly unique prefixes.
    """
    if repetitions < 1 or warmups_per_condition < 0:
        raise ValueError("repetitions must be positive and warmups cannot be negative")
    conditions = list(CacheCondition)
    samples: dict[CacheCondition, list[RequestProfile]] = {item: [] for item in conditions}

    for condition in conditions:
        for index in range(warmups_per_condition):
            await run_request(_cache_request(condition, index, warmup=True))

    rng = random.Random(seed)
    for repetition in range(repetitions):
        ordered = conditions.copy()
        rng.shuffle(ordered)
        for condition in ordered:
            profile = await run_request(_cache_request(condition, repetition, warmup=False))
            samples[condition].append(profile)

    results = []
    for condition in conditions:
        profiles = samples[condition]
        successful = [p for p in profiles if p.status is RequestStatus.SUCCEEDED]
        hit_values = [p.cache_hit for p in successful if p.cache_hit is not None]
        memory = [p.memory_after_mb for p in successful if p.memory_after_mb is not None]
        results.append(
            CacheConditionResult(
                condition=condition,
                samples=len(profiles),
                failures=len(profiles) - len(successful),
                p50_latency_ms=percentile((p.total_latency_ms for p in successful), 0.50),
                p95_latency_ms=percentile((p.total_latency_ms for p in successful), 0.95),
                mean_output_tokens_per_second=mean_optional(
                    p.output_tokens_per_second for p in successful
                ),
                peak_memory_mb=max(memory) if memory else None,
                cache_hit_rate=(sum(hit_values) / len(hit_values)) if hit_values else None,
            )
        )
    return CacheExperimentReport(
        warmups_per_condition=warmups_per_condition,
        repetitions=repetitions,
        randomized_seed=seed,
        conditions=results,
        interpretation=(
            "Repeated byte-identical prefixes can reuse prefill KV state. Unique prefixes test "
            "lookup overhead without reuse; short prefixes may not amortize that overhead."
        ),
    )


def _cache_request(condition: CacheCondition, sample_index: int, *, warmup: bool) -> CacheRequest:
    repeated = "optimo-cache-experiment-byte-identical-prefix-v1"
    prefix = (
        f"optimo-cache-experiment-unique-{sample_index}-warmup-{int(warmup)}"
        if condition is CacheCondition.ENABLED_UNIQUE_PREFIX
        else repeated
    )
    return CacheRequest(
        condition=condition,
        prefix_cache_enabled=condition is not CacheCondition.DISABLED,
        prefix=prefix,
        sample_index=sample_index,
        warmup=warmup,
    )
