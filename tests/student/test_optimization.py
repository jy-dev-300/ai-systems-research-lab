from optimization.models import ExperimentResult, InferenceConfiguration
from optimization.search import pareto_frontier, search_configurations


def config(name):
    return InferenceConfiguration(
        name=name,
        backend="mock",
        model="mock",
        max_model_length=128,
    )


def result(name, success, latency, throughput, memory=100, cost=1):
    return ExperimentResult(
        configuration=config(name),
        task_success_rate=success,
        p95_latency_ms=latency,
        throughput_requests_per_second=throughput,
        peak_gpu_memory_mb=memory,
        cost_per_task=cost,
    )


def test_search_runs_grid_and_rejects_below_quality() -> None:
    measured = {
        "good": result("good", 1.0, 10, 5),
        "bad": result("bad", 0.5, 1, 50),
    }
    accepted = search_configurations(
        [config("good"), config("bad")],
        0.9,
        lambda candidate: measured[candidate.name],
    )
    assert [item.configuration.name for item in accepted] == ["good"]


def test_pareto_frontier_handles_dominated_and_tied_results() -> None:
    strong = result("strong", 1.0, 10, 10)
    dominated = result("dominated", 0.9, 20, 5)
    tied = result("tied", 1.0, 10, 10)
    frontier = pareto_frontier([strong, dominated, tied])
    assert [item.configuration.name for item in frontier] == ["strong", "tied"]


def test_missing_memory_is_worse_when_other_objectives_are_equal() -> None:
    measured = result("measured", 1.0, 10, 10, memory=100)
    missing = result("missing", 1.0, 10, 10, memory=None)
    assert pareto_frontier([missing, measured]) == [measured]
