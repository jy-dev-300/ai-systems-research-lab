from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from harness.adapters.registry import adapter_registry, register_builtin_adapters
from harness.analysis.bottlenecks import analyze_bottlenecks
from harness.build.agent import BuildAgent
from harness.build.backend import ScriptedBuildBackend
from harness.build.loop import BuildLoop, review_build
from harness.build.models import BuildResult, BuildStatus, BuildTask
from harness.experiments.runner import ExperimentRunner
from harness.experiments.sweep import load_sweep
from harness.knowledge import ExperimentKnowledgeStore
from harness.models import ExperimentConfig, ExperimentResult
from harness.programs.classifier import JevWorkflowClassifier
from harness.programs.runner import ProgramRunner
from harness.research.advisor import ResearchAdvisor
from harness.research.ingestion import load_local_documents
from harness.selection import (
    FixedHarnessPolicy,
    HistoricalAggregatePolicy,
    compare_selection_policies,
)


def _result_path(run_id: str, root: Path = Path(".harness/runs")) -> Path:
    path = root / run_id / "result.json"
    if not path.exists():
        raise FileNotFoundError(f"Run not found: {run_id} ({path})")
    return path


def _load_result(run_id: str) -> ExperimentResult:
    return ExperimentResult.model_validate_json(_result_path(run_id).read_text(encoding="utf-8"))


async def _run_config(path: str, *, all_tasks: bool, task_id: str | None) -> int:
    if Path(path).suffix.casefold() == ".md":
        result = await ProgramRunner(JevWorkflowClassifier()).run(path)
        print(result.model_dump_json(indent=2))
        return 0 if result.status.value == "completed" else 2
    config = ExperimentConfig.from_file(path)
    register_builtin_adapters()
    adapter = adapter_registry.create(config.adapter.name)
    tasks = adapter.load_tasks(config)
    if task_id:
        tasks = [task for task in tasks if str(task.get("id", task.get("instance_id"))) == task_id]
        if not tasks:
            raise ValueError(f"Task {task_id!r} not found")
    elif not all_tasks:
        tasks = tasks[:1]
    results = await ExperimentRunner().run_batch(config, tasks)
    print(
        json.dumps(
            [
                {"run_id": item.run_id, "task_id": item.task_id, "success": item.success}
                for item in results
            ],
            indent=2,
        )
    )
    return 0 if all(item.success for item in results) else 2


async def _run_sweep(path: str) -> int:
    outcomes: list[dict[str, Any]] = []
    runner = ExperimentRunner()
    for config in load_sweep(path):
        results = await runner.run_batch(config)
        outcomes.extend(
            {"config": config.fingerprint(), "run_id": item.run_id, "success": item.success}
            for item in results
        )
    print(json.dumps(outcomes, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Universal AI experimentation harness")
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run one task from a config")
    run.add_argument("config")
    run.add_argument("--task-id")

    batch = subparsers.add_parser("batch", help="run every task from a config")
    batch.add_argument("config")

    sweep = subparsers.add_parser("sweep", help="run a parameter grid")
    sweep.add_argument("config")

    analyze = subparsers.add_parser("analyze", help="generate a bottleneck report")
    analyze.add_argument("run_id")

    research = subparsers.add_parser("research", help="propose controlled experiments")
    research.add_argument("run_id")
    research.add_argument("--documents", default="research/fixtures/research_documents.json")

    mlflow = subparsers.add_parser("mlflow", help="show or start the local MLflow UI")
    mlflow.add_argument("--backend-store-uri", default="./mlruns")
    mlflow.add_argument("--port", type=int, default=5000)
    mlflow.add_argument("--execute", action="store_true")

    select = subparsers.add_parser(
        "compare-selection", help="compare fixed and historical harness choices"
    )
    select.add_argument("config")
    select.add_argument("--knowledge", default=".harness/knowledge.sqlite3")
    select.add_argument("--fixed", default="simple_loop")

    build = subparsers.add_parser("build", help="run an approved BuildTask")
    build.add_argument("task")

    review = subparsers.add_parser("review-build", help="accept or reject a completed build")
    review.add_argument("build_id")
    review.add_argument("decision", choices=("accept", "reject"))
    review.add_argument("--root", default=".harness/builds")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    if args.command == "run":
        return asyncio.run(_run_config(args.config, all_tasks=False, task_id=args.task_id))
    if args.command == "batch":
        return asyncio.run(_run_config(args.config, all_tasks=True, task_id=None))
    if args.command == "sweep":
        return asyncio.run(_run_sweep(args.config))
    if args.command == "analyze":
        report = analyze_bottlenecks(_load_result(args.run_id))
        print(report.model_dump_json(indent=2))
        return 0
    if args.command == "research":
        result = _load_result(args.run_id)
        report = analyze_bottlenecks(result)
        advisor = ResearchAdvisor(load_local_documents(args.documents))
        proposals = advisor.propose(result, report)
        print(json.dumps([item.model_dump(mode="json") for item in proposals], indent=2))
        return 0
    if args.command == "mlflow":
        command = [
            "mlflow",
            "ui",
            "--backend-store-uri",
            args.backend_store_uri,
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
        ]
        if args.execute:
            return subprocess.run(command, check=False).returncode
        print(" ".join(command))
        return 0
    if args.command == "compare-selection":
        config = ExperimentConfig.from_file(args.config)
        register_builtin_adapters()
        tasks = adapter_registry.create(config.adapter.name).load_tasks(config)
        normalized = [
            {**task, "task_id": task.get("id", task.get("instance_id"))} for task in tasks
        ]
        comparison = compare_selection_policies(
            domain=config.experiment.domain,
            tasks=normalized,
            fixed=FixedHarnessPolicy(args.fixed),
            adaptive=HistoricalAggregatePolicy(ExperimentKnowledgeStore(args.knowledge)),
        )
        print(json.dumps(comparison, indent=2))
        return 0
    if args.command == "build":
        task = BuildTask.from_file(args.task)
        backend = ScriptedBuildBackend.from_task(task)
        result = asyncio.run(BuildLoop(BuildAgent(backend)).run(task))
        print(
            json.dumps(
                {
                    "build_id": result.build_id,
                    "status": result.final_status.value,
                    "approval_state": result.approval_state.value,
                    "criteria": [
                        item.model_dump(mode="json") for item in result.acceptance_criteria_results
                    ],
                },
                indent=2,
            )
        )
        return 0 if result.final_status is BuildStatus.SUCCESS else 2
    if args.command == "review-build":
        path = Path(args.root) / args.build_id / "build_result.json"
        result = BuildResult.model_validate_json(path.read_text(encoding="utf-8"))
        reviewed = review_build(result, accept=args.decision == "accept")
        path.write_text(reviewed.model_dump_json(indent=2), encoding="utf-8")
        print(
            json.dumps(
                {"build_id": reviewed.build_id, "approval_state": reviewed.approval_state.value},
                indent=2,
            )
        )
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
