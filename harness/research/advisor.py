from __future__ import annotations

from harness.analysis.bottlenecks import BottleneckReport
from harness.models import ExperimentResult
from harness.research.models import CandidateExperiment, ResearchDocument, ResearchEvidence
from harness.research.retrieval import HybridResearchRetriever

TECHNIQUES = {
    "low_retrieval_recall": (
        "reciprocal rank fusion with dense and lexical retrieval",
        "Combine complementary rank signals without comparing incompatible raw scores.",
        ["recall_at_k", "mrr", "ndcg"],
    ),
    "retrieval_dominates_latency": (
        "two-stage retrieval with a smaller candidate pool",
        "Reduce expensive reranker work while holding the first-stage retriever constant.",
        ["retrieval_latency_ms", "recall_at_k"],
    ),
    "excessive_model_calls": (
        "prefix caching and stricter termination",
        "Reuse invariant prefixes and cap redundant turns.",
        ["model_calls", "latency_ms", "cost_usd"],
    ),
    "excessive_retries": (
        "failure-conditioned retry policy",
        "Retry only transient failures and terminate repeated deterministic failures.",
        ["retries", "latency_ms", "success"],
    ),
    "tool_execution_dominates_latency": (
        "bounded tool concurrency",
        "Execute independent tool calls concurrently while preserving trace ordering.",
        ["latency_ms", "tool_calls"],
    ),
    "environment_failure_after_action": (
        "closed-loop replanning from state feedback",
        "Replan after observed state transitions rather than committing to an open-loop sequence.",
        ["task_success", "replans", "steps"],
    ),
}


class ResearchAdvisor:
    def __init__(self, documents: list[ResearchDocument]) -> None:
        self.retriever = HybridResearchRetriever(documents)

    def propose(
        self, result: ExperimentResult, report: BottleneckReport, *, limit: int = 3
    ) -> list[CandidateExperiment]:
        candidates: list[CandidateExperiment] = []
        for observation in report.observations:
            if observation.kind == "no_dominant_bottleneck":
                continue
            technique, mechanism, metrics = TECHNIQUES.get(
                observation.kind,
                (
                    "single-variable ablation",
                    "Change one relevant parameter and retain the current run as control.",
                    observation.affected_metrics or ["success"],
                ),
            )
            sources = self.retriever.search(f"{observation.kind} {technique}", limit=3)
            evidence = [
                ResearchEvidence(
                    document_id=document.id,
                    title=document.title,
                    url=document.url,
                    evidence_type=document.evidence_type,
                    relevance_score=score,
                )
                for document, score in sources
            ]
            candidates.append(
                CandidateExperiment(
                    observed_bottleneck=observation.kind,
                    candidate_technique=technique,
                    supporting_sources=evidence,
                    mechanism=mechanism,
                    expected_metric_affected=metrics,
                    assumptions=["External evidence may not transfer to this workload."],
                    controlled_experiment=(
                        f"Hold task set, seed, model, and {result.harness_type} constant; compare "
                        f"the current config against one config enabling {technique}."
                    ),
                    implementation_reference=next(
                        (doc.implementation_url for doc, _ in sources if doc.implementation_url),
                        None,
                    ),
                    evidence_level="external_evidence_only",
                    confidence=min(0.85, 0.4 + 0.1 * len(evidence)),
                )
            )
            if len(candidates) >= limit:
                break
        return candidates
