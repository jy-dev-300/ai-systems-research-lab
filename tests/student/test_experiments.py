import pytest

from benchmarks.cache_experiment import CacheCondition, run_cache_experiment
from benchmarks.concurrency_experiment import run_concurrency_experiment
from benchmarks.quantization_experiment import run_quantization_experiment
from harnesses.contracts import HarnessResult, RunStatus
from optimization.models import InferenceConfiguration
from profiling.models import RequestProfile, RequestStatus

pytestmark = pytest.mark.student


def profile(request_id="sample", latency=10, cache_hit=None):
    return RequestProfile(
        request_id=request_id,
        status=RequestStatus.SUCCEEDED,
        time_to_first_token_ms=None,
        total_latency_ms=latency,
        queue_time_ms=1,
        input_tokens=5,
        output_tokens=5,
        output_tokens_per_second=500,
        memory_before_mb=100,
        memory_after_mb=110,
        cache_hit=cache_hit,
    )


@pytest.mark.anyio
async def test_cache_experiment_warms_up_and_reuses_exact_prefix() -> None:
    requests = []

    async def run(request):
        requests.append(request)
        return profile(cache_hit=request.condition is CacheCondition.ENABLED_REPEATED_PREFIX)

    report = await run_cache_experiment(run, repetitions=3, warmups_per_condition=1)
    assert len(requests) == 12
    repeated = [
        request.prefix
        for request in requests
        if request.condition is CacheCondition.ENABLED_REPEATED_PREFIX
    ]
    assert len(set(repeated)) == 1
    assert report.conditions[2].cache_hit_rate == 1


@pytest.mark.anyio
async def test_concurrency_experiment_holds_workload_constant(addition_task) -> None:
    seen = []

    async def run(configuration, task, request_id):
        seen.append((configuration.name, task.id))
        return profile(request_id)

    configs = [
        InferenceConfiguration(
            name=name,
            backend="mock",
            model="mock",
            max_model_length=128,
            concurrency=2,
            arrival_pattern="burst",
        )
        for name in ("one", "two")
    ]
    reports = await run_concurrency_experiment(configs, [addition_task], run, repetitions=2)
    assert [report.requests for report in reports] == [2, 2]
    assert seen.count(("one", addition_task.id)) == seen.count(("two", addition_task.id))


@pytest.mark.anyio
async def test_concurrency_experiment_records_oom_instead_of_aborting(addition_task) -> None:
    async def run(configuration, task, request_id):
        raise MemoryError("CUDA OOM")

    configuration = InferenceConfiguration(
        name="oom",
        backend="mock",
        model="mock",
        max_model_length=128,
        arrival_pattern="burst",
    )
    report = await run_concurrency_experiment([configuration], [addition_task], run, repetitions=1)
    assert report[0].failures == 1
    assert report[0].oom_failures == 1


@pytest.mark.anyio
async def test_quantization_quality_gate_rejects_faster_wrong_model(addition_task) -> None:
    configs = [
        InferenceConfiguration(
            name="accurate",
            backend="mock",
            model="artifact-a",
            quantization="bf16",
            max_model_length=128,
        ),
        InferenceConfiguration(
            name="fast-wrong",
            backend="mock",
            model="artifact-b",
            quantization="int4-awq",
            max_model_length=128,
            extra={"flags": ["--quantization", "awq"]},
        ),
    ]

    async def run(configuration, task):
        correct = configuration.name == "accurate"
        result = HarnessResult(
            task.id,
            RunStatus.SUCCEEDED,
            task.expected_answer if correct else "wrong",
            1,
            0,
            (),
        )
        return result, profile(latency=20 if correct else 5)

    report = await run_quantization_experiment(configs, [addition_task], run)
    assert report.eligible_configuration_names == ["accurate"]
