import pytest

from profiling.models import RequestStatus
from profiling.profiler import InferenceProfiler, calculate_bottlenecks
from serving.models import InferenceMetrics, ModelAction, ModelRequest, ModelResponse, Usage
from tracing.events import TraceEvent, TraceEventType

pytestmark = pytest.mark.student


class ProfileBackend:
    name = "profile"

    async def generate(self, request):
        return ModelResponse(
            action=ModelAction(kind="final", final_answer="ok"),
            usage=Usage(input_tokens=5, output_tokens=10),
            metrics=InferenceMetrics(total_latency_ms=100, queue_time_ms=20),
        )


@pytest.mark.anyio
async def test_profiler_uses_generation_duration_for_token_rate() -> None:
    profile = await InferenceProfiler(ProfileBackend()).profile(ModelRequest(messages=[]))
    assert profile.status is RequestStatus.SUCCEEDED
    assert profile.time_to_first_token_ms is None
    assert profile.generation_duration_ms == 80
    assert profile.output_tokens_per_second == 125


def test_bottlenecks_are_non_overlapping_and_reconcile() -> None:
    events = (
        TraceEvent(
            run_id="r",
            task_id="t",
            sequence=0,
            type=TraceEventType.MODEL_CALL,
            duration_ms=100,
            payload={"queue_time_ms": 20},
        ),
        TraceEvent(
            run_id="r",
            task_id="t",
            sequence=1,
            type=TraceEventType.TOOL_RESULT,
            duration_ms=10,
        ),
        TraceEvent(
            run_id="r",
            task_id="t",
            sequence=2,
            type=TraceEventType.MODEL_CALL,
            duration_ms=50,
        ),
        TraceEvent(
            run_id="r",
            task_id="t",
            sequence=3,
            type=TraceEventType.TERMINATION,
            payload={"total_elapsed_ms": 175},
        ),
    )
    result = calculate_bottlenecks(events)
    assert result.model_inference_ms == 80
    assert result.repeated_model_call_ms == 50
    assert result.queueing_ms == 20
    assert result.tool_execution_ms == 10
    assert result.harness_overhead_ms == 15
