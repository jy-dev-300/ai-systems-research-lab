from __future__ import annotations

from harness.adapters.base import AdapterEvaluation, AdapterRunOutput
from harness.models import FailureClassification, FailureType


def classify_failure(
    *,
    error: Exception | None,
    evaluation: AdapterEvaluation | None,
    run_output: AdapterRunOutput | None,
) -> FailureClassification | None:
    """Classify conservatively from concrete error/evaluation evidence."""
    if error is not None:
        message = str(error)
        lowered = message.casefold()
        if isinstance(error, TimeoutError) or "timeout" in lowered:
            kind = FailureType.TIMEOUT
        elif isinstance(error, (FileNotFoundError, ImportError)) or any(
            term in lowered for term in ("requires:", "not installed", "checkout is empty")
        ):
            kind = FailureType.RESOURCE
        elif isinstance(error, (ValueError, KeyError)):
            kind = FailureType.CONFIGURATION
        elif "model" in lowered or "backend" in lowered:
            kind = FailureType.MODEL
        else:
            kind = FailureType.UNKNOWN
        return FailureClassification(
            type=kind,
            subtype=type(error).__name__,
            evidence=[message or repr(error)],
            confidence=0.9,
        )
    if evaluation is None or evaluation.success:
        return None
    subtype = evaluation.failure_subtype
    if subtype and any(word in subtype for word in ("retriev", "chunk", "rerank", "evidence")):
        kind = FailureType.RETRIEVAL
    elif subtype and any(word in subtype for word in ("test", "environment", "action")):
        kind = FailureType.EXECUTION
    elif subtype and "plan" in subtype:
        kind = FailureType.PLANNING
    else:
        kind = FailureType.EVALUATION
    return FailureClassification(
        type=kind,
        subtype=subtype,
        evidence=evaluation.evidence,
        confidence=0.8 if subtype else 0.5,
    )
