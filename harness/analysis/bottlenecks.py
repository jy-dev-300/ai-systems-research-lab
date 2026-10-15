from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from harness.models import ExperimentResult, TraceEvent


class BottleneckObservation(BaseModel):
    kind: str
    severity: Literal["low", "medium", "high"]
    evidence: list[str]
    affected_metrics: list[str] = Field(default_factory=list)


class BottleneckReport(BaseModel):
    run_id: str
    observations: list[BottleneckObservation] = Field(default_factory=list)
    thresholds: dict[str, float] = Field(default_factory=dict)


def analyze_bottlenecks(
    result: ExperimentResult,
    trace: list[TraceEvent] | None = None,
    *,
    latency_ms: float = 2_000,
    model_calls: int = 8,
    retries: int = 2,
) -> BottleneckReport:
    events = trace if trace is not None else result.trace
    observations: list[BottleneckObservation] = []

    def add(
        kind: str, severity: Literal["low", "medium", "high"], evidence: str, *metrics: str
    ) -> None:
        observations.append(
            BottleneckObservation(
                kind=kind,
                severity=severity,
                evidence=[evidence],
                affected_metrics=list(metrics),
            )
        )

    if result.latency_ms > latency_ms:
        add(
            "high_total_latency",
            "high",
            f"{result.latency_ms:.1f}ms > {latency_ms:.1f}ms",
            "latency_ms",
        )
    if result.model_calls > model_calls:
        add(
            "excessive_model_calls",
            "high",
            f"{result.model_calls} > {model_calls}",
            "model_calls",
            "cost_usd",
        )
    if result.retries > retries:
        add("excessive_retries", "high", f"{result.retries} > {retries}", "retries", "latency_ms")
    recall = result.domain_metrics.get("recall_at_k")
    if isinstance(recall, (int, float)) and recall < 0.8:
        add("low_retrieval_recall", "high", f"recall_at_k={float(recall):.3f}", "recall_at_k")
    retrieval_ms = next((m.value for m in result.metrics if m.name == "retrieval_latency_ms"), 0.0)
    if result.latency_ms and retrieval_ms / result.latency_ms > 0.6:
        add(
            "retrieval_dominates_latency",
            "medium",
            f"retrieval share={retrieval_ms / result.latency_ms:.1%}",
            "retrieval_latency_ms",
        )
    tool_ms = sum(event.duration_ms or 0 for event in events if event.type == "tool_result")
    if result.latency_ms and tool_ms / result.latency_ms > 0.6:
        add(
            "tool_execution_dominates_latency",
            "medium",
            f"tool share={tool_ms / result.latency_ms:.1%}",
            "latency_ms",
        )
    if result.cost_usd > 0 and result.output_tokens > result.input_tokens * 2:
        add(
            "generation_dominates_cost",
            "medium",
            "output tokens exceed 2x input tokens",
            "cost_usd",
            "output_tokens",
        )
    if result.failure and result.failure.subtype == "environment_goal_not_reached":
        add(
            "environment_failure_after_action",
            "high",
            "; ".join(result.failure.evidence),
            "task_success",
        )
    if not observations:
        add("no_dominant_bottleneck", "low", "No configured evidence threshold was crossed")
    return BottleneckReport(
        run_id=result.run_id,
        observations=observations,
        thresholds={
            "latency_ms": latency_ms,
            "model_calls": float(model_calls),
            "retries": float(retries),
        },
    )
