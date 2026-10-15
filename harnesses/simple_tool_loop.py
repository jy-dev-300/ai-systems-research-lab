from __future__ import annotations

import json
from dataclasses import dataclass
from time import perf_counter

from benchmarks.models import BenchmarkTask
from harnesses.contracts import HarnessConfig, HarnessResult, HarnessState, RunStatus
from serving.base import ServingBackend
from serving.models import ChatMessage, ModelRequest
from tools.registry import ToolRegistry
from tracing.events import TraceEventType
from tracing.recorder import TraceRecorder


@dataclass
class SimpleToolLoop:
    backend: ServingBackend
    tools: ToolRegistry
    tracer: TraceRecorder
    config: HarnessConfig = HarnessConfig()

    async def run(self, task: BenchmarkTask) -> HarnessResult:
        """Execute explicit model/tool turns until a terminal condition."""
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

        # These fields script the local deterministic fixture. Real serving
        # backends must decide what to do from the conversation itself.
        request_metadata = {}
        if self.backend.name == "deterministic-mock":
            request_metadata = {
                "expected_tool_calls": [call.model_dump() for call in task.expected_tool_calls],
                "expected_answer": task.expected_answer,
            }

        while state.step_count < self.config.max_steps:
            request = ModelRequest(
                messages=list(state.messages),
                tools=self.tools.definitions(),
                metadata=request_metadata,
            )
            state.step_count += 1
            model_calls += 1
            call_started = perf_counter()
            try:
                response = await self.backend.generate(request)
            except Exception as error:
                state.status = RunStatus.FAILED
                self.tracer.record(
                    TraceEventType.ERROR,
                    {
                        "phase": "model_call",
                        "error_type": type(error).__name__,
                        "message": str(error),
                    },
                    duration_ms=(perf_counter() - call_started) * 1_000,
                )
                self.tracer.record(
                    TraceEventType.TERMINATION,
                    {
                        "status": state.status.value,
                        "total_elapsed_ms": (perf_counter() - started) * 1_000,
                    },
                )
                return HarnessResult(
                    state.task_id,
                    state.status,
                    None,
                    model_calls,
                    tool_calls,
                    self.tracer.snapshot(),
                )
            action = response.action
            state.seen_actions.append(action.model_dump_json())
            self.tracer.record(
                TraceEventType.MODEL_CALL,
                {
                    "step": state.step_count,
                    "backend": self.backend.name,
                    "action": action.model_dump(),
                    "usage": response.usage.model_dump(),
                    "queue_time_ms": response.metrics.queue_time_ms,
                },
                duration_ms=response.metrics.total_latency_ms,
            )

            if action.kind == "final":
                state.final_answer = action.final_answer
                state.status = RunStatus.SUCCEEDED
                self.tracer.record(
                    TraceEventType.TERMINATION,
                    {
                        "status": state.status.value,
                        "answer": state.final_answer,
                        "total_elapsed_ms": (perf_counter() - started) * 1_000,
                    },
                )
                return HarnessResult(
                    task_id=state.task_id,
                    status=state.status,
                    final_answer=state.final_answer,
                    model_calls=model_calls,
                    tool_calls=tool_calls,
                    trace=self.tracer.snapshot(),
                )

            call = action.tool_call
            # ModelAction validates this pairing, but keep the boundary explicit:
            # model-produced structured data is never authority to run a tool.
            if call is None:
                state.status = RunStatus.FAILED
                self.tracer.record(
                    TraceEventType.ERROR,
                    {"reason": "tool action did not contain a tool call"},
                )
                self.tracer.record(
                    TraceEventType.TERMINATION,
                    {
                        "status": state.status.value,
                        "total_elapsed_ms": (perf_counter() - started) * 1_000,
                    },
                )
                return HarnessResult(
                    state.task_id,
                    state.status,
                    None,
                    model_calls,
                    tool_calls,
                    self.tracer.snapshot(),
                )

            self.tracer.record(TraceEventType.TOOL_CALL, call.model_dump())
            tool_started = perf_counter()
            try:
                result = self.tools.execute(call.name, call.arguments)
            except Exception as error:
                state.status = RunStatus.FAILED
                self.tracer.record(
                    TraceEventType.ERROR,
                    {
                        "tool_call_id": call.id,
                        "name": call.name,
                        "error_type": type(error).__name__,
                        "message": str(error),
                    },
                    duration_ms=(perf_counter() - tool_started) * 1_000,
                )
                self.tracer.record(
                    TraceEventType.TERMINATION,
                    {
                        "status": state.status.value,
                        "total_elapsed_ms": (perf_counter() - started) * 1_000,
                    },
                )
                return HarnessResult(
                    state.task_id,
                    state.status,
                    None,
                    model_calls,
                    tool_calls,
                    self.tracer.snapshot(),
                )

            tool_calls += 1
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

        state.status = RunStatus.MAX_STEPS
        self.tracer.record(
            TraceEventType.TERMINATION,
            {
                "status": state.status.value,
                "max_steps": self.config.max_steps,
                "total_elapsed_ms": (perf_counter() - started) * 1_000,
            },
        )
        return HarnessResult(
            task_id=state.task_id,
            status=state.status,
            final_answer=None,
            model_calls=model_calls,
            tool_calls=tool_calls,
            trace=self.tracer.snapshot(),
        )
