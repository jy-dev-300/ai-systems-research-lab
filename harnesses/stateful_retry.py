from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from time import perf_counter

from pydantic import ValidationError

from benchmarks.models import BenchmarkTask
from harnesses.contracts import HarnessConfig, HarnessResult, HarnessState, RunStatus
from harnesses.retry import (
    FailureDisposition,
    FailureKind,
    RetryContext,
    RetryPolicy,
)
from serving.base import ServingBackend
from serving.models import ChatMessage, ModelRequest
from tools.registry import InvalidToolArgumentsError, ToolRegistry, UnknownToolError
from tracing.events import TraceEventType
from tracing.recorder import TraceRecorder


@dataclass
class StatefulRetryHarness:
    """Agent loop with explicit recovery policy and loop detection."""

    backend: ServingBackend
    tools: ToolRegistry
    tracer: TraceRecorder
    retry_policy: RetryPolicy
    config: HarnessConfig = HarnessConfig()

    async def run(self, task: BenchmarkTask) -> HarnessResult:
        started = perf_counter()
        state = HarnessState(
            task_id=task.id,
            messages=[
                ChatMessage(
                    role="system",
                    content="Use tools when required. Return a concise answer.",
                ),
                ChatMessage(role="user", content=task.prompt),
            ],
        )
        model_calls = 0
        tool_calls = 0
        action_counts: dict[str, int] = {}
        metadata = {}
        if self.backend.name == "deterministic-mock":
            metadata = {
                "expected_tool_calls": [call.model_dump() for call in task.expected_tool_calls],
                "expected_answer": task.expected_answer,
            }

        def finish(status: RunStatus, answer: str | None = None) -> HarnessResult:
            state.status = status
            state.final_answer = answer
            self.tracer.record(
                TraceEventType.TERMINATION,
                {
                    "status": status.value,
                    "answer": answer,
                    "total_elapsed_ms": (perf_counter() - started) * 1_000,
                    "retries": state.retry_count,
                },
            )
            return HarnessResult(
                task_id=state.task_id,
                status=status,
                final_answer=answer,
                model_calls=model_calls,
                tool_calls=tool_calls,
                trace=self.tracer.snapshot(),
            )

        async def recover(failure: FailureKind, message: str, repeats: int = 0) -> bool:
            decision = self.retry_policy.decide(
                RetryContext(
                    failure=failure,
                    attempts=state.retry_count,
                    max_retries=self.config.max_retries,
                    repeated_action_count=repeats,
                )
            )
            self.tracer.record(
                TraceEventType.RETRY,
                {
                    "failure": failure.value,
                    "disposition": decision.disposition.value,
                    "reason": decision.reason,
                    "attempt": state.retry_count,
                },
                duration_ms=decision.backoff_seconds * 1_000,
            )
            if decision.disposition is FailureDisposition.TERMINATE:
                return False
            state.retry_count += 1
            if decision.disposition is FailureDisposition.FALLBACK:
                state.messages.append(
                    ChatMessage(
                        role="user",
                        content=(
                            f"The previous action failed: {message}. "
                            "Choose a different valid action."
                        ),
                    )
                )
            if decision.backoff_seconds:
                await asyncio.sleep(decision.backoff_seconds)
            return True

        while state.step_count < self.config.max_steps:
            request = ModelRequest(
                messages=list(state.messages),
                tools=self.tools.definitions(),
                metadata=metadata,
            )
            call_started = perf_counter()
            state.step_count += 1
            model_calls += 1
            try:
                response = await self.backend.generate(request)
            except TimeoutError as error:
                self.tracer.record(
                    TraceEventType.ERROR,
                    {"failure": FailureKind.MODEL_TIMEOUT.value, "message": str(error)},
                    duration_ms=(perf_counter() - call_started) * 1_000,
                )
                if await recover(FailureKind.MODEL_TIMEOUT, str(error)):
                    continue
                return finish(RunStatus.FAILED)
            except (ValidationError, ValueError, TypeError) as error:
                self.tracer.record(
                    TraceEventType.ERROR,
                    {"failure": FailureKind.MALFORMED_MODEL_OUTPUT.value, "message": str(error)},
                    duration_ms=(perf_counter() - call_started) * 1_000,
                )
                if await recover(FailureKind.MALFORMED_MODEL_OUTPUT, str(error)):
                    continue
                return finish(RunStatus.FAILED)

            action = response.action
            action_key = action.model_dump_json()
            action_counts[action_key] = action_counts.get(action_key, 0) + 1
            repeats = action_counts[action_key]
            state.seen_actions.append(action_key)
            self.tracer.record(
                TraceEventType.MODEL_CALL,
                {"step": state.step_count, "action": action.model_dump()},
                duration_ms=response.metrics.total_latency_ms,
            )

            if repeats >= 2:
                failure = FailureKind.LOOP_DETECTED if repeats >= 3 else FailureKind.REPEATED_ACTION
                if await recover(failure, "the action was repeated", repeats):
                    continue
                return finish(RunStatus.FAILED)

            if action.kind == "final":
                return finish(RunStatus.SUCCEEDED, action.final_answer)

            call = action.tool_call
            if call is None:
                if await recover(
                    FailureKind.MALFORMED_MODEL_OUTPUT,
                    "tool action omitted its tool call",
                ):
                    continue
                return finish(RunStatus.FAILED)

            tool_calls += 1
            self.tracer.record(TraceEventType.TOOL_CALL, call.model_dump())
            tool_started = perf_counter()
            try:
                result = self.tools.execute(call.name, call.arguments)
            except (UnknownToolError, InvalidToolArgumentsError) as error:
                failure = (
                    FailureKind.UNKNOWN_TOOL
                    if isinstance(error, UnknownToolError)
                    else FailureKind.INVALID_TOOL_ARGUMENTS
                )
                self.tracer.record(
                    TraceEventType.ERROR,
                    {
                        "failure": failure.value,
                        "name": call.name,
                        "message": str(error),
                    },
                    duration_ms=(perf_counter() - tool_started) * 1_000,
                )
                if await recover(failure, str(error)):
                    continue
                return finish(RunStatus.FAILED)
            except Exception as error:
                failure = FailureKind.TOOL_EXCEPTION
                self.tracer.record(
                    TraceEventType.ERROR,
                    {
                        "failure": failure.value,
                        "name": call.name,
                        "error_type": type(error).__name__,
                        "message": str(error),
                    },
                    duration_ms=(perf_counter() - tool_started) * 1_000,
                )
                if await recover(failure, str(error)):
                    continue
                return finish(RunStatus.FAILED)

            self.tracer.record(
                TraceEventType.TOOL_RESULT,
                {"tool_call_id": call.id, "name": call.name, "result": result},
                duration_ms=(perf_counter() - tool_started) * 1_000,
            )
            state.messages.extend(
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
                        content=json.dumps(result, default=str),
                    ),
                ]
            )

        return finish(RunStatus.MAX_STEPS)
