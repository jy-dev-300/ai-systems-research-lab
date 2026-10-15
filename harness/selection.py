from __future__ import annotations

from typing import Any, Protocol

from harness.knowledge import ExperimentKnowledgeStore


class HarnessSelectionPolicy(Protocol):
    def select(self, *, domain: str, task_metadata: dict[str, Any]) -> str: ...


class FixedHarnessPolicy:
    def __init__(self, harness_type: str = "simple_loop") -> None:
        self.harness_type = harness_type

    def select(self, *, domain: str, task_metadata: dict[str, Any]) -> str:
        return self.harness_type


class HistoricalAggregatePolicy:
    """Select the best observed harness; this is empirical routing, not optimality."""

    def __init__(self, store: ExperimentKnowledgeStore, fallback: str = "simple_loop") -> None:
        self.store = store
        self.fallback = fallback

    def select(self, *, domain: str, task_metadata: dict[str, Any]) -> str:
        task_id = task_metadata.get("task_id")
        summary = self.store.summarize(domain=domain, task_id=str(task_id) if task_id else None)
        configs = summary["configurations"]
        if not configs:
            return self.fallback
        return max(
            configs,
            key=lambda name: (
                configs[name]["success_rate"],
                configs[name]["mean_quality"],
                -configs[name]["runs"],
            ),
        )


def compare_selection_policies(
    *,
    domain: str,
    tasks: list[dict[str, Any]],
    fixed: HarnessSelectionPolicy,
    adaptive: HarnessSelectionPolicy,
) -> list[dict[str, str]]:
    return [
        {
            "task_id": str(task.get("task_id", task.get("id", "unknown"))),
            "fixed": fixed.select(domain=domain, task_metadata=task),
            "task_conditioned": adaptive.select(domain=domain, task_metadata=task),
        }
        for task in tasks
    ]
