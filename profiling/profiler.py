from __future__ import annotations

from time import perf_counter
from uuid import uuid4

from profiling.models import BottleneckBreakdown, RequestProfile, RequestStatus
from serving.base import ServingBackend
from serving.models import ModelRequest
from tracing.events import TraceEvent, TraceEventType


class InferenceProfiler:
    """Measure a request without inventing unavailable backend data."""

    def __init__(self, backend: ServingBackend) -> None:
        self.backend = backend

    async def profile(self, request: ModelRequest) -> RequestProfile:
        request_id = str(request.metadata.get("request_id") or uuid4())
        started = perf_counter()
        try:
            response = await self.backend.generate(request)
        except TimeoutError as error:
            return RequestProfile(
                request_id=request_id,
                status=RequestStatus.TIMED_OUT,
                time_to_first_token_ms=None,
                total_latency_ms=(perf_counter() - started) * 1_000,
                queue_time_ms=None,
                input_tokens=0,
                output_tokens=0,
                output_tokens_per_second=None,
                memory_before_mb=None,
                memory_after_mb=None,
                error=str(error),
            )
        except Exception as error:
            return RequestProfile(
                request_id=request_id,
                status=RequestStatus.FAILED,
                time_to_first_token_ms=None,
                total_latency_ms=(perf_counter() - started) * 1_000,
                queue_time_ms=None,
                input_tokens=0,
                output_tokens=0,
                output_tokens_per_second=None,
                memory_before_mb=None,
                memory_after_mb=None,
                error=f"{type(error).__name__}: {error}",
            )

        wall_ms = (perf_counter() - started) * 1_000
        metrics = response.metrics
        # Backend total latency covers queue + generation. Subtract supported
        # queue time so throughput is decode/generation based, not wall based.
        generation_ms = metrics.total_latency_ms
        if metrics.queue_time_ms is not None:
            generation_ms = max(0.0, generation_ms - metrics.queue_time_ms)
        tokens_per_second = None
        if response.usage.output_tokens > 0 and generation_ms > 0:
            tokens_per_second = response.usage.output_tokens / (generation_ms / 1_000)
        return RequestProfile(
            request_id=request_id,
            status=RequestStatus.SUCCEEDED,
            time_to_first_token_ms=metrics.time_to_first_token_ms,
            total_latency_ms=wall_ms,
            queue_time_ms=metrics.queue_time_ms,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            output_tokens_per_second=tokens_per_second,
            memory_before_mb=metrics.gpu_memory_before_mb,
            memory_after_mb=metrics.gpu_memory_after_mb,
            cache_hit=metrics.cache_hit,
            generation_duration_ms=generation_ms if generation_ms > 0 else None,
        )


def calculate_bottlenecks(events: tuple[TraceEvent, ...]) -> BottleneckBreakdown:
    """Return non-overlapping categories that reconcile to recorded wall time.

    Queue time is removed from each model duration. The first model generation
    is inference; later generations are repeated-model-call cost. Tool and retry
    durations are direct event durations. Remaining wall time is harness overhead.
    """
    model_events = [event for event in events if event.type is TraceEventType.MODEL_CALL]
    model_inference = 0.0
    repeated_model = 0.0
    queueing = 0.0
    for index, event in enumerate(model_events):
        duration = event.duration_ms or 0.0
        queue = float(event.payload.get("queue_time_ms") or 0.0)
        queue = min(max(queue, 0.0), duration)
        generation = max(0.0, duration - queue)
        queueing += queue
        if index == 0:
            model_inference += generation
        else:
            repeated_model += generation

    tool_execution = sum(
        event.duration_ms or 0.0 for event in events if event.type is TraceEventType.TOOL_RESULT
    )
    retry = sum(event.duration_ms or 0.0 for event in events if event.type is TraceEventType.RETRY)
    recorded_total = next(
        (
            float(event.payload["total_elapsed_ms"])
            for event in reversed(events)
            if event.type is TraceEventType.TERMINATION
            and event.payload.get("total_elapsed_ms") is not None
        ),
        0.0,
    )
    accounted = model_inference + repeated_model + queueing + tool_execution + retry
    harness_overhead = max(0.0, recorded_total - accounted)
    return BottleneckBreakdown(
        model_inference_ms=model_inference,
        queueing_ms=queueing,
        tool_execution_ms=tool_execution,
        retry_ms=retry,
        repeated_model_call_ms=repeated_model,
        harness_overhead_ms=harness_overhead,
    )
