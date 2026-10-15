import pytest

from tracing.events import TraceEventType
from tracing.recorder import TraceRecorder

pytestmark = pytest.mark.student


def test_trace_recorder_orders_events_and_preserves_identity() -> None:
    recorder = TraceRecorder("run-1", "task-1")
    first = recorder.record(TraceEventType.MODEL_CALL, {"model": "mock"}, 2.5)
    second = recorder.record(TraceEventType.TERMINATION, {"status": "succeeded"})
    assert first.sequence == 0
    assert second.sequence == 1
    assert all(event.run_id == "run-1" for event in recorder.snapshot())
    assert first.duration_ms == 2.5


def test_trace_recorder_redacts_and_bounds_payloads() -> None:
    recorder = TraceRecorder("run", "task")
    event = recorder.record(
        TraceEventType.ERROR,
        {"api_key": "do-not-store", "provider_payload": "x" * 5_000},
    )
    assert event.payload["api_key"] == "<redacted>"
    assert len(event.payload["provider_payload"]) < 4_200
