from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from serving.models import ChatMessage


class RunStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    MAX_STEPS = "max_steps"


@dataclass(frozen=True)
class HarnessConfig:
    max_steps: int = 6
    max_retries: int = 2

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")


@dataclass
class HarnessState:
    task_id: str
    messages: list[ChatMessage]
    step_count: int = 0
    retry_count: int = 0
    status: RunStatus = RunStatus.RUNNING
    final_answer: str | None = None
    seen_actions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HarnessResult:
    task_id: str
    status: RunStatus
    final_answer: str | None
    model_calls: int
    tool_calls: int
    trace: tuple[Any, ...]


class Harness(Protocol):
    """A model-independent agent control loop."""

    async def run(self, task: Any) -> HarnessResult:
        """Run one task to a terminal state and return its reconstructable trajectory."""
        ...
