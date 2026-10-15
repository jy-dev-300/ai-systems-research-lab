from __future__ import annotations

from uuid import uuid4

from benchmarks.models import BenchmarkTask
from harnesses.contracts import HarnessResult, RunStatus
from serving.base import ServingBackend
from serving.models import ChatMessage, ModelRequest
from tools.registry import ToolRegistry
from tracing.events import TraceEvent, TraceEventType


class TemporaryBaselineRunner:
    """Legacy fixed runner retained for compatibility with baseline comparisons.

    It handles exactly one tool call with two explicit model calls.
    """

    def __init__(self, backend: ServingBackend, tools: ToolRegistry) -> None:
        self.backend = backend
        self.tools = tools

    async def run(self, task: BenchmarkTask) -> HarnessResult:
        run_id = str(uuid4())
        events: list[TraceEvent] = []
        messages = [
            ChatMessage(role="system", content="Use tools when required. Return a concise answer."),
            ChatMessage(role="user", content=task.prompt),
        ]
        metadata = {
            "expected_tool_calls": [call.model_dump() for call in task.expected_tool_calls],
            "expected_answer": task.expected_answer,
        }

        first = await self.backend.generate(
            ModelRequest(messages=messages, tools=self.tools.definitions(), metadata=metadata)
        )
        events.append(
            self._event(
                run_id,
                task.id,
                events,
                TraceEventType.MODEL_CALL,
                {"action": first.action.model_dump(), "usage": first.usage.model_dump()},
            )
        )
        if first.action.kind != "tool" or first.action.tool_call is None:
            return HarnessResult(task.id, RunStatus.FAILED, None, 1, 0, tuple(events))

        call = first.action.tool_call
        events.append(
            self._event(run_id, task.id, events, TraceEventType.TOOL_CALL, call.model_dump())
        )
        result = self.tools.execute(call.name, call.arguments)
        events.append(
            self._event(
                run_id,
                task.id,
                events,
                TraceEventType.TOOL_RESULT,
                {"tool_call_id": call.id, "name": call.name, "result": result},
            )
        )
        messages.extend(
            [
                ChatMessage(role="assistant", content=f"Requested tool {call.name}"),
                ChatMessage(role="tool", name=call.name, tool_call_id=call.id, content=str(result)),
            ]
        )

        second = await self.backend.generate(
            ModelRequest(messages=messages, tools=self.tools.definitions(), metadata=metadata)
        )
        events.append(
            self._event(
                run_id,
                task.id,
                events,
                TraceEventType.MODEL_CALL,
                {"action": second.action.model_dump(), "usage": second.usage.model_dump()},
            )
        )
        succeeded = second.action.kind == "final" and second.action.final_answer is not None
        answer = second.action.final_answer if succeeded else None
        events.append(
            self._event(
                run_id,
                task.id,
                events,
                TraceEventType.TERMINATION,
                {"status": "succeeded" if succeeded else "failed", "answer": answer},
            )
        )
        return HarnessResult(
            task_id=task.id,
            status=RunStatus.SUCCEEDED if succeeded else RunStatus.FAILED,
            final_answer=answer,
            model_calls=2,
            tool_calls=1,
            trace=tuple(events),
        )

    @staticmethod
    def _event(
        run_id: str,
        task_id: str,
        events: list[TraceEvent],
        event_type: TraceEventType,
        payload: dict,
    ) -> TraceEvent:
        return TraceEvent(
            run_id=run_id,
            task_id=task_id,
            sequence=len(events),
            type=event_type,
            payload=payload,
        )
