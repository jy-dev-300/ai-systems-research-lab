from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.schemas import (
    LabActivityView,
    LabExperimentView,
    LabMetricView,
    LabOpportunityView,
    LabOverviewResponse,
    LabSystemView,
    ResearchDocumentView,
)
from harness.adapters.registry import adapter_registry, register_builtin_adapters
from harness.analysis.bottlenecks import analyze_bottlenecks
from harness.models import ExperimentResult
from harness.programs.registry import ProgramRegistry

SYSTEM_ORDER = ("bo", "myvoice", "swebench", "maniskill", "harness")
DOMAIN_IDS = {
    "bo": "bo",
    "myvoice": "myvoice",
    "swe-bench": "swebench",
    "swebench": "swebench",
    "physical-ai": "maniskill",
    "maniskill": "maniskill",
    "harness": "harness",
}
BOTTLENECK_TITLES = {
    "high_total_latency": "End-to-end latency exceeds the research threshold",
    "retrieval_dominates_latency": "Retrieval dominates query latency",
    "tool_execution_dominates_latency": "Tool execution dominates task latency",
    "excessive_model_calls": "Agent is making excessive model calls",
    "excessive_retries": "Retries are consuming the execution budget",
    "low_retrieval_recall": "Relevant evidence is missing from retrieval",
    "generation_dominates_cost": "Generation dominates experiment cost",
    "environment_failure_after_action": "Environment fails after agent actions",
}


def _read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None


def _records(root: Path, directory: str, filename: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    base = root / directory
    if not base.is_dir():
        return records
    for path in base.glob(f"*/{filename}"):
        payload = _read_json(path)
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _timestamp(payload: dict[str, Any]) -> str:
    return str(payload.get("completed_at") or payload.get("timestamp") or "")


def _system_id(domain: str) -> str:
    normalized = domain.casefold().strip()
    return DOMAIN_IDS.get(normalized, normalized.replace(" ", "-"))


def _metrics(payload: dict[str, Any], *, limit: int = 8) -> list[LabMetricView]:
    values: list[LabMetricView] = []
    seen: set[str] = set()
    for item in payload.get("metrics", []):
        if not isinstance(item, dict) or not isinstance(item.get("value"), (int, float)):
            continue
        name = str(item.get("name", "metric"))
        values.append(LabMetricView(name=name, value=float(item["value"]), unit=item.get("unit")))
        seen.add(name)
    common = (
        ("quality", payload.get("quality"), None),
        ("latency_ms", payload.get("latency_ms"), "ms"),
        ("cost_usd", payload.get("cost_usd"), "USD"),
    )
    for name, value, unit in common:
        if name not in seen and isinstance(value, (int, float)):
            values.append(LabMetricView(name=name, value=float(value), unit=unit))
    return values[:limit]


def _experiment(payload: dict[str, Any]) -> LabExperimentView | None:
    try:
        config = payload.get("config", {})
        identity = config.get("experiment", {}) if isinstance(config, dict) else {}
        return LabExperimentView(
            run_id=str(payload["run_id"]),
            task_id=str(payload.get("task_id", "unknown")),
            domain=str(payload.get("domain", "Unknown")),
            name=str(identity.get("name", payload.get("domain", "Experiment"))),
            timestamp=payload["timestamp"],
            success=bool(payload.get("success", False)),
            is_mock=str(payload.get("provider", "")).casefold() == "mock"
            or "mock" in str(payload.get("backend", "")).casefold(),
            harness_type=str(payload.get("harness_type", "unknown")),
            model=str(payload.get("model", "unknown")),
            latency_ms=float(payload.get("latency_ms", 0)),
            cost_usd=float(payload.get("cost_usd", 0)),
            metrics=_metrics(payload),
            failure=payload.get("failure"),
            trace=[item for item in payload.get("trace", []) if isinstance(item, dict)][:100],
            artifacts=[item for item in payload.get("artifacts", []) if isinstance(item, dict)],
            output=payload.get("output", {}) if isinstance(payload.get("output", {}), dict) else {},
            config=config if isinstance(config, dict) else {},
            error=str(payload["error"]) if payload.get("error") else None,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _systems(root: Path, runs: list[dict[str, Any]]) -> list[LabSystemView]:
    register_builtin_adapters()
    registry = ProgramRegistry(root)
    definitions = {definition.adapter: definition for definition in registry.definitions()}
    systems: list[LabSystemView] = []
    for adapter_name in SYSTEM_ORDER:
        definition = definitions.get(adapter_name)
        if definition is None:
            continue
        repo = registry.repo_path(definition)
        adapter = adapter_registry.create(adapter_name, repo=repo)
        knowledge = adapter.project_knowledge()
        matching = [run for run in runs if _system_id(str(run.get("domain", ""))) == adapter_name]
        matching.sort(key=_timestamp, reverse=True)
        latest = matching[0] if matching else None
        if not repo.is_dir():
            status = "unavailable"
        elif latest is None:
            status = "ready"
        elif latest.get("success"):
            status = "healthy"
        else:
            status = "attention"
        systems.append(
            LabSystemView(
                id=adapter_name,
                name=knowledge.project_name,
                project_type=knowledge.project_type,
                summary=knowledge.summary,
                status=status,
                repo=str(repo),
                repo_exists=repo.is_dir(),
                capabilities=sorted(item.value for item in knowledge.capabilities),
                metrics=_metrics(latest, limit=4) if latest else [],
                run_count=len(matching),
                success_count=sum(bool(run.get("success")) for run in matching),
                last_run_at=_timestamp(latest) if latest else None,
            )
        )
    return systems


def _opportunities(
    runs: list[dict[str, Any]], programs: list[dict[str, Any]]
) -> list[LabOpportunityView]:
    opportunities: list[LabOpportunityView] = []
    seen: set[tuple[str, str]] = set()
    for payload in sorted(runs, key=_timestamp, reverse=True):
        try:
            result = ExperimentResult.model_validate(payload)
        except (TypeError, ValueError):
            continue
        system_id = _system_id(result.domain)
        for observation in analyze_bottlenecks(result).observations:
            if observation.kind == "no_dominant_bottleneck":
                continue
            key = (system_id, observation.kind)
            if key in seen:
                continue
            seen.add(key)
            opportunities.append(
                LabOpportunityView(
                    id=f"{result.run_id}:{observation.kind}",
                    system_id=system_id,
                    title=BOTTLENECK_TITLES.get(
                        observation.kind, observation.kind.replace("_", " ").title()
                    ),
                    description="; ".join(observation.evidence),
                    severity=observation.severity,
                    status="observed",
                    run_id=result.run_id,
                    affected_metrics=observation.affected_metrics,
                    evidence=observation.evidence,
                )
            )
    for program in sorted(programs, key=_timestamp, reverse=True):
        outputs = program.get("outputs", {})
        candidates = outputs.get("candidate_experiments", []) if isinstance(outputs, dict) else []
        system_id = _system_id(str(program.get("project", "")))
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            candidate_id = str(candidate.get("id", "candidate"))
            opportunities.append(
                LabOpportunityView(
                    id=f"candidate:{candidate_id}",
                    system_id=system_id,
                    title=str(candidate.get("candidate_technique", "Candidate experiment")),
                    description=str(candidate.get("mechanism", "External evidence to review.")),
                    severity="medium",
                    status="candidate",
                    affected_metrics=[
                        str(item) for item in candidate.get("expected_metric_affected", [])
                    ],
                    evidence=[str(item) for item in candidate.get("assumptions", [])],
                    candidate_experiment_id=candidate_id,
                )
            )
    return opportunities


def _research_documents(root: Path) -> list[ResearchDocumentView]:
    payload = _read_json(root / "research/fixtures/research_documents.json")
    if not isinstance(payload, list):
        return []
    documents: list[ResearchDocumentView] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            documents.append(ResearchDocumentView.model_validate(item))
        except ValueError:
            continue
    return sorted(documents, key=lambda item: item.publication_date or "", reverse=True)


def _activity(
    programs: list[dict[str, Any]], builds: list[dict[str, Any]]
) -> list[LabActivityView]:
    activity: list[LabActivityView] = []
    for payload in programs:
        activity.append(
            LabActivityView(
                id=str(payload.get("program_run_id", "program")),
                kind="program",
                title=f"{payload.get('project') or 'Lab'} program",
                status=str(payload.get("status", "unknown")),
                timestamp=payload.get("completed_at") or payload.get("started_at"),
                summary=str(payload.get("summary", "Program completed.")),
                details={
                    "workflow": (payload.get("decision") or {}).get("workflow"),
                    "experiment_run_ids": payload.get("experiment_run_ids", []),
                },
            )
        )
    for payload in builds:
        task = payload.get("task", {}) if isinstance(payload.get("task"), dict) else {}
        activity.append(
            LabActivityView(
                id=str(payload.get("build_id", "build")),
                kind="build",
                title=str(task.get("goal", "Build task")),
                status=str(payload.get("final_status", "unknown")),
                timestamp=payload.get("completed_at") or payload.get("created_at"),
                summary=(
                    f"{len(payload.get('files_changed', []))} files changed; "
                    f"{len(payload.get('iterations', []))} iterations"
                ),
                details={
                    "target_project": task.get("target_project"),
                    "unresolved_issues": payload.get("unresolved_issues", []),
                },
            )
        )
    return sorted(activity, key=lambda item: str(item.timestamp or ""), reverse=True)


def build_lab_overview(root: str | Path | None = None) -> LabOverviewResponse:
    lab_root = Path(root or os.getenv("LAB_ROOT") or Path(__file__).resolve().parents[1]).resolve()
    runs = _records(lab_root, ".harness/runs", "result.json")
    runs.sort(key=_timestamp, reverse=True)
    programs = _records(lab_root, ".harness/programs", "program_result.json")
    builds = _records(lab_root, ".harness/builds", "build_result.json")
    experiments = [item for payload in runs if (item := _experiment(payload)) is not None]
    documents = _research_documents(lab_root)
    opportunities = _opportunities(runs, programs)
    activity = _activity(programs, builds)
    return LabOverviewResponse(
        generated_at=datetime.now(UTC),
        artifact_root=str(lab_root / ".harness"),
        mlflow_url=os.getenv("MLFLOW_UI_URL", "http://127.0.0.1:5000"),
        systems=_systems(lab_root, runs),
        opportunities=opportunities,
        experiments=experiments,
        research_documents=documents,
        activity=activity,
        counts={
            "systems": len(SYSTEM_ORDER),
            "experiments": len(runs),
            "successful_experiments": sum(bool(run.get("success")) for run in runs),
            "opportunities": len(opportunities),
            "papers": len(documents),
            "builds": len(builds),
            "programs": len(programs),
            "candidates": sum(item.status == "candidate" for item in opportunities),
        },
    )
