from __future__ import annotations

import re
from collections.abc import Mapping
from time import perf_counter
from typing import Any, Protocol

from harness.programs.models import Workflow, WorkflowContext, WorkflowDecision

PROMPT_VERSION = "workflow-intent-v1"
WORKFLOW_CAPABILITIES: dict[Workflow, tuple[str, ...]] = {
    Workflow.BUILD: ("inspect", "build", "test"),
    Workflow.EVAL: ("inspect", "evaluate"),
    Workflow.RESEARCH: ("inspect", "build", "test", "evaluate", "experiment"),
}


class WorkflowClassifier(Protocol):
    def classify(self, context: WorkflowContext) -> WorkflowDecision: ...


class JevWorkflowClassifier:
    """One typed JEV choice; orchestration remains outside this class."""

    name = "Jev"

    def __init__(
        self,
        *,
        model: str = "jev-latest",
        client: Any | None = None,
        question: Any | None = None,
        version: str = "typesafe-sdk>=0.7.4,<1",
    ) -> None:
        self.model = model
        self.client = client
        self.question = question
        self.version = version

    def _client(self) -> Any:
        if self.client is not None:
            return self.client
        try:
            from typesafe_sdk import TypeSafeClient
        except ImportError as error:
            raise RuntimeError(
                "JEV routing requires typesafe-sdk; install: pip install -e '.[jev]'"
            ) from error
        self.client = TypeSafeClient()
        return self.client

    def _question(self) -> Any:
        if self.question is not None:
            return self.question
        try:
            from typesafe_sdk import Choice
        except ImportError as error:
            raise RuntimeError(
                "JEV routing requires typesafe-sdk; install: pip install -e '.[jev]'"
            ) from error
        self.question = Choice(
            instructions=(
                "Classify the primary top-level workflow requested by the researcher. "
                "Internal testing does not make BUILD into EVAL, and implementation inside a "
                "hypothesis-driven investigation does not make RESEARCH into BUILD."
            ),
            criteria={
                "BUILD": "Implement, modify, integrate, fix, or construct a system.",
                "EVAL": "Measure, compare, benchmark, inspect, or evaluate without modification.",
                "RESEARCH": (
                    "Discover or validate an improvement through a hypothesis and controlled "
                    "comparison; this may include building and evaluation."
                ),
            },
        )
        return self.question

    def classify(self, context: WorkflowContext) -> WorkflowDecision:
        started = perf_counter()
        response = self._client().system_one(
            state={
                "program": context.program.contents,
                "project": context.project,
                "adapter": context.adapter,
                "available_capabilities": list(context.capabilities),
                "current_state": context.project_state.summary,
                "project_knowledge": context.project_knowledge,
            },
            questions={"workflow": self._question()},
            model=self.model,
        )
        answer = response.choices["workflow"]
        workflow = Workflow(str(answer.choice).upper())
        probabilities = {
            str(key).upper(): float(value)
            for key, value in getattr(answer, "probabilities", {}).items()
        }
        return WorkflowDecision(
            workflow=workflow,
            confidence=float(answer.confidence),
            reason=(
                f"JEV selected {workflow.value} as the primary workflow "
                f"(confidence {float(answer.confidence):.3f})."
            ),
            required_capabilities=WORKFLOW_CAPABILITIES[workflow],
            classifier=self.name,
            classifier_version=self.version,
            model=self.model,
            prompt_version=PROMPT_VERSION,
            probabilities=probabilities,
            latency_ms=(perf_counter() - started) * 1_000,
        )


class RuleWorkflowClassifier:
    """Deterministic baseline and offline smoke-test classifier."""

    name = "rules"
    version = "workflow-rules-v1"

    def classify(self, context: WorkflowContext) -> WorkflowDecision:
        started = perf_counter()
        text = context.program.contents.casefold()
        text = text.replace("do not modify", "").replace("without modifying", "")
        research = (
            r"\binvestigat(?:e|es|ed|ing)\b",
            r"\bhypothesis\b",
            r"\bfind a way\b",
            r"\bworth adding\b",
            r"\bwithout increasing\b",
            r"\bwhether\b",
            r"\bkeep improvements\b",
        )
        build = (
            r"\bimplement(?:ed|ing)?\b",
            r"\bfix(?:es|ing)?\b",
            r"\badd\b",
            r"\bintegrat(?:e|es|ed|ing)\b",
            r"\bbuild\b",
            r"\bcreat(?:e|es|ed|ing)\b",
            r"\bmodif(?:y|ies|ied|ying)\b",
        )
        evaluate = (
            r"\bbenchmark(?:s|ed|ing)?\b",
            r"\bcompar(?:e|es|ed|ing)\b",
            r"\bmeasur(?:e|es|ed|ing)\b",
            r"\bevaluat(?:e|es|ed|ing)\b",
            r"\binspect(?:s|ed|ing)?\b",
        )
        scores = {
            Workflow.RESEARCH: sum(bool(re.search(term, text)) for term in research),
            Workflow.BUILD: sum(bool(re.search(term, text)) for term in build),
            Workflow.EVAL: sum(bool(re.search(term, text)) for term in evaluate),
        }
        workflow = max(scores, key=lambda item: (scores[item], item is Workflow.RESEARCH))
        ordered = sorted(scores.values(), reverse=True)
        confidence = (
            0.5
            if ordered[0] == 0 or ordered[0] == ordered[1]
            else min(0.99, 0.72 + 0.08 * (ordered[0] - ordered[1]))
        )
        total = sum(scores.values()) or 1
        probabilities = {item.value: score / total for item, score in scores.items()}
        return WorkflowDecision(
            workflow=workflow,
            confidence=confidence,
            reason=f"Rule baseline matched {scores[workflow]} {workflow.value} intent signals.",
            required_capabilities=WORKFLOW_CAPABILITIES[workflow],
            classifier=self.name,
            classifier_version=self.version,
            model="deterministic",
            prompt_version=PROMPT_VERSION,
            probabilities=probabilities,
            latency_ms=(perf_counter() - started) * 1_000,
        )


def validate_capabilities(
    decision: WorkflowDecision, available: tuple[str, ...]
) -> tuple[str, ...]:
    available_set = set(available)
    return tuple(item for item in decision.required_capabilities if item not in available_set)


def choice_payload(decision: WorkflowDecision) -> Mapping[str, object]:
    return decision.model_dump(mode="json")
