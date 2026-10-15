from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from time import perf_counter

from benchmarks.models import BenchmarkTask
from harnesses.contracts import HarnessConfig, HarnessResult, RunStatus
from serving.base import ServingBackend
from serving.models import ChatMessage, ModelRequest
from tools.registry import ToolRegistry
from tracing.events import TraceEventType
from tracing.recorder import TraceRecorder


class PlanStepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PlanStep:
    description: str
    tool_name: str | None = None
    status: PlanStepStatus = PlanStepStatus.PENDING
    result: str | None = None


@dataclass
class Plan:
    steps: list[PlanStep] = field(default_factory=list)
    revision: int = 0


@dataclass
class PlannerExecutorHarness:
    """Planner, executor, evaluator, and replanning control flow."""

    backend: ServingBackend
    tools: ToolRegistry
    tracer: TraceRecorder
    config: HarnessConfig = HarnessConfig()

    async def run(self, task: BenchmarkTask) -> HarnessResult:
        started = perf_counter()
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "Plan internally, execute one step at a time with tools, evaluate the result, "
                    "and return a concise final answer."
                ),
            ),
            ChatMessage(role="user", content=task.prompt),
        ]
        # Expected calls may script a deterministic fixture, but never enter a
        # real provider request.
        if self.backend.name == "deterministic-mock":
            plan = Plan(
                steps=[
                    PlanStep(description=f"Call {call.name}", tool_name=call.name)
                    for call in task.expected_tool_calls
                ]
            )
            metadata = {
                "expected_tool_calls": [call.model_dump() for call in task.expected_tool_calls],
                "expected_answer": task.expected_answer,
            }
        else:
            plan = Plan(steps=[PlanStep(description="Solve the task using the available tools")])
            metadata = {}

        self.tracer.record(
            TraceEventType.STATE_UPDATE,
            {"phase": "planning", "revision": plan.revision, "steps": self._plan_payload(plan)},
        )
        model_calls = 0
        tool_calls = 0
        active_index = 0

        def finish(status: RunStatus, answer: str | None = None) -> HarnessResult:
            self.tracer.record(
                TraceEventType.TERMINATION,
                {
                    "status": status.value,
                    "answer": answer,
                    "plan": self._plan_payload(plan),
                    "total_elapsed_ms": (perf_counter() - started) * 1_000,
                },
            )
            return HarnessResult(
                task.id,
                status,
                answer,
                model_calls,
                tool_calls,
                self.tracer.snapshot(),
            )

        while model_calls < self.config.max_steps:
            response = await self.backend.generate(
                ModelRequest(
                    messages=list(messages),
                    tools=self.tools.definitions(),
                    metadata=metadata,
                )
            )
            model_calls += 1
            action = response.action
            self.tracer.record(
                TraceEventType.MODEL_CALL,
                {
                    "phase": "execution",
                    "step": model_calls,
                    "action": action.model_dump(),
                    "usage": response.usage.model_dump(),
                    "queue_time_ms": response.metrics.queue_time_ms,
                },
                duration_ms=response.metrics.total_latency_ms,
            )

            if action.kind == "final":
                pending = [step for step in plan.steps if step.status is PlanStepStatus.PENDING]
                self.tracer.record(
                    TraceEventType.STATE_UPDATE,
                    {
                        "phase": "evaluation",
                        "complete": not pending,
                        "pending_steps": len(pending),
                    },
                )
                if pending and self.backend.name == "deterministic-mock":
                    return finish(RunStatus.FAILED)
                for step in pending:
                    step.status = PlanStepStatus.COMPLETED
                return finish(RunStatus.SUCCEEDED, action.final_answer)

            call = action.tool_call
            if call is None:
                return finish(RunStatus.FAILED)

            if active_index >= len(plan.steps):
                plan.revision += 1
                plan.steps.append(
                    PlanStep(description=f"Replanned call to {call.name}", tool_name=call.name)
                )
                self.tracer.record(
                    TraceEventType.STATE_UPDATE,
                    {
                        "phase": "replanning",
                        "reason": "model proposed an additional required action",
                        "revision": plan.revision,
                        "steps": self._plan_payload(plan),
                    },
                )

            step = plan.steps[active_index]
            if step.tool_name is not None and step.tool_name != call.name:
                plan.revision += 1
                step.description = f"Replanned call to {call.name}"
                step.tool_name = call.name
                self.tracer.record(
                    TraceEventType.STATE_UPDATE,
                    {
                        "phase": "replanning",
                        "reason": "model selected a different tool",
                        "revision": plan.revision,
                    },
                )

            step.status = PlanStepStatus.RUNNING
            tool_calls += 1
            self.tracer.record(
                TraceEventType.TOOL_CALL,
                {**call.model_dump(), "plan_step": active_index},
            )
            tool_started = perf_counter()
            try:
                result = self.tools.execute(call.name, call.arguments)
            except Exception as error:
                step.status = PlanStepStatus.FAILED
                step.result = str(error)
                self.tracer.record(
                    TraceEventType.ERROR,
                    {
                        "phase": "execution",
                        "plan_step": active_index,
                        "error_type": type(error).__name__,
                        "message": str(error),
                    },
                    duration_ms=(perf_counter() - tool_started) * 1_000,
                )
                return finish(RunStatus.FAILED)

            step.status = PlanStepStatus.COMPLETED
            step.result = json.dumps(result, default=str)
            self.tracer.record(
                TraceEventType.TOOL_RESULT,
                {
                    "tool_call_id": call.id,
                    "name": call.name,
                    "result": result,
                    "plan_step": active_index,
                },
                duration_ms=(perf_counter() - tool_started) * 1_000,
            )
            self.tracer.record(
                TraceEventType.STATE_UPDATE,
                {
                    "phase": "evaluation",
                    "plan_step": active_index,
                    "status": step.status.value,
                },
            )
            messages.extend(
                [
                    ChatMessage(
                        role="assistant",
                        content=f"Requested tool {call.name}",
                        tool_call_id=call.id,
                    ),
                    ChatMessage(
                        role="tool",
                        name=call.name,
                        tool_call_id=call.id,
                        content=step.result,
                    ),
                ]
            )
            active_index += 1

        return finish(RunStatus.MAX_STEPS)

    @staticmethod
    def _plan_payload(plan: Plan) -> list[dict[str, str | None]]:
        return [
            {
                "description": step.description,
                "tool_name": step.tool_name,
                "status": step.status.value,
                "result": step.result,
            }
            for step in plan.steps
        ]
