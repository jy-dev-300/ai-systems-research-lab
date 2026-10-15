from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from harness.programs.classifier import WorkflowClassifier
from harness.programs.models import (
    ProgramDocument,
    ProjectState,
    Workflow,
    WorkflowContext,
)


class RoutingEvaluation(BaseModel):
    dataset_version: str
    classifier: str
    examples: int
    accuracy: float
    per_class: dict[str, dict[str, float]]
    confusion_matrix: dict[str, dict[str, int]]
    mean_brier_score: float
    mean_latency_ms: float
    failures: list[dict[str, Any]] = Field(default_factory=list)


def evaluate_classifier(
    classifier: WorkflowClassifier, dataset_path: str | Path
) -> RoutingEvaluation:
    payload = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    cases = payload["cases"]
    matrix = {gold.value: {pred.value: 0 for pred in Workflow} for gold in Workflow}
    tp: defaultdict[Workflow, int] = defaultdict(int)
    fp: defaultdict[Workflow, int] = defaultdict(int)
    fn: defaultdict[Workflow, int] = defaultdict(int)
    failures = []
    brier = latency = 0.0
    name = "unknown"
    for index, case in enumerate(cases):
        contents = str(case["program"])
        program = ProgramDocument(
            path=f"eval:{index}",
            filename=f"{case.get('project', 'bo').casefold()}-{index}.md",
            contents=contents,
            sha256=__import__("hashlib").sha256(contents.encode()).hexdigest(),
        )
        context = WorkflowContext(
            program=program,
            project=str(case.get("project", "Bo")),
            adapter=str(case.get("adapter", "bo")),
            capabilities=("inspect", "build", "test", "evaluate", "experiment", "research"),
            project_state=ProjectState(
                repo="eval", exists=True, summary=str(case.get("state", ""))
            ),
        )
        decision = classifier.classify(context)
        name = decision.classifier
        gold = Workflow(case["gold"])
        predicted = decision.workflow
        matrix[gold.value][predicted.value] += 1
        latency += decision.latency_ms
        for label in Workflow:
            probability = decision.probabilities.get(
                label.value, 1.0 if label is predicted else 0.0
            )
            brier += (probability - float(label is gold)) ** 2 / len(Workflow)
        if predicted is gold:
            tp[gold] += 1
        else:
            fp[predicted] += 1
            fn[gold] += 1
            failures.append(
                {
                    "id": case["id"],
                    "program": contents,
                    "gold": gold.value,
                    "predicted": predicted.value,
                    "confidence": decision.confidence,
                }
            )
    total = len(cases)
    per_class = {}
    for label in Workflow:
        precision = tp[label] / max(1, tp[label] + fp[label])
        recall = tp[label] / max(1, tp[label] + fn[label])
        per_class[label.value] = {"precision": precision, "recall": recall}
    return RoutingEvaluation(
        dataset_version=str(payload["version"]),
        classifier=name,
        examples=total,
        accuracy=(total - len(failures)) / max(1, total),
        per_class=per_class,
        confusion_matrix=matrix,
        mean_brier_score=brier / max(1, total),
        mean_latency_ms=latency / max(1, total),
        failures=failures,
    )
