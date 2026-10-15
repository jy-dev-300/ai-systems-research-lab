from __future__ import annotations

from collections.abc import Callable
from typing import Any

from harness.models import ExperimentConfig
from harnesses.contracts import Harness, HarnessConfig
from harnesses.planner_executor import PlannerExecutorHarness
from harnesses.simple_tool_loop import SimpleToolLoop
from serving.mock import DeterministicMockBackend
from serving.vllm import VLLMBackend
from tools.registry import default_tool_registry
from tracing.recorder import TraceRecorder

HarnessFactory = Callable[[ExperimentConfig, str, str], Harness]


class HarnessRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, HarnessFactory] = {}

    def register(self, name: str, factory: HarnessFactory, *, replace: bool = False) -> None:
        if name in self._factories and not replace:
            raise ValueError(f"Harness already registered: {name}")
        self._factories[name] = factory

    def create(self, name: str, config: ExperimentConfig, run_id: str, task_id: str) -> Harness:
        try:
            return self._factories[name](config, run_id, task_id)
        except KeyError as error:
            choices = ", ".join(sorted(self._factories)) or "none"
            raise ValueError(f"Unknown harness {name!r}; available: {choices}") from error

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


def _backend(config: ExperimentConfig) -> Any:
    if config.model.backend == "deterministic":
        return DeterministicMockBackend()
    if config.model.backend == "vllm":
        if not config.model.base_url:
            raise ValueError("model.base_url is required for the vllm backend")
        return VLLMBackend(config.model.base_url, config.model.name)
    raise ValueError(f"Unsupported model backend: {config.model.backend}")


def _simple(config: ExperimentConfig, run_id: str, task_id: str) -> Harness:
    return SimpleToolLoop(
        backend=_backend(config),
        tools=default_tool_registry(),
        tracer=TraceRecorder(run_id, task_id),
        config=HarnessConfig(
            max_steps=config.harness.max_steps,
            max_retries=config.harness.max_retries,
        ),
    )


def _planner(config: ExperimentConfig, run_id: str, task_id: str) -> Harness:
    return PlannerExecutorHarness(
        backend=_backend(config),
        tools=default_tool_registry(),
        tracer=TraceRecorder(run_id, task_id),
        config=HarnessConfig(
            max_steps=config.harness.max_steps,
            max_retries=config.harness.max_retries,
        ),
    )


harness_registry = HarnessRegistry()
harness_registry.register("simple_loop", _simple)
harness_registry.register("planner_executor", _planner)
