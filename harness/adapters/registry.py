from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from harness.adapters.base import TaskAdapter

AdapterFactory = Callable[[], TaskAdapter]


class AdapterRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, AdapterFactory] = {}

    def register(self, name: str, factory: AdapterFactory, *, replace: bool = False) -> None:
        if name in self._factories and not replace:
            raise ValueError(f"Adapter already registered: {name}")
        self._factories[name] = factory

    def create(self, name: str, *, repo: str | Path | None = None) -> TaskAdapter:
        try:
            adapter = self._factories[name]()
        except KeyError as error:
            available = ", ".join(sorted(self._factories)) or "none"
            raise ValueError(f"Unknown adapter {name!r}; available: {available}") from error
        return adapter.bind_repo(repo) if repo is not None else adapter

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


adapter_registry = AdapterRegistry()


def register_builtin_adapters() -> None:
    from harness.adapters.bo import BoAdapter
    from harness.adapters.harness import HarnessAdapter
    from harness.adapters.maniskill import ManiSkillAdapter
    from harness.adapters.myvoice import MyVoiceAdapter
    from harness.adapters.swebench import SWEbenchAdapter

    builtins: dict[str, AdapterFactory] = {
        "bo": BoAdapter,
        "harness": HarnessAdapter,
        "maniskill": ManiSkillAdapter,
        "myvoice": MyVoiceAdapter,
        "swebench": SWEbenchAdapter,
    }
    for name, factory in builtins.items():
        if name not in adapter_registry.names():
            adapter_registry.register(name, factory)
