from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from harness.adapters.base import AdapterEvaluation, AdapterRunOutput, TaskAdapter
from harness.adapters.common import resolve_workspace_path
from harness.adapters.registry import AdapterRegistry, adapter_registry, register_builtin_adapters
from harness.failures import classify_failure
from harness.harness_registry import HarnessRegistry, harness_registry
from harness.knowledge import ExperimentKnowledgeStore
from harness.models import Artifact, ExperimentConfig, ExperimentResult, TraceEvent
from harness.tracking import ExperimentTracker, build_tracker

logger = logging.getLogger("harness.experiments")


def _git_commit(path: Path) -> str | None:
    try:
        process = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return process.stdout.strip() if process.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _task_id(task: dict[str, Any]) -> str:
    for key in ("id", "instance_id", "task_id", "query_id"):
        if key in task:
            return str(task[key])
    raise ValueError("Task must contain id, instance_id, task_id, or query_id")


class ExperimentRunner:
    def __init__(
        self,
        *,
        adapters: AdapterRegistry = adapter_registry,
        harnesses: HarnessRegistry = harness_registry,
        tracker: ExperimentTracker | None = None,
        knowledge: ExperimentKnowledgeStore | None = None,
    ) -> None:
        register_builtin_adapters()
        self.adapters = adapters
        self.harnesses = harnesses
        self.tracker = tracker
        self.knowledge = knowledge

    def _adapter(self, config: ExperimentConfig) -> TaskAdapter:
        adapter = self.adapters.create(config.adapter.name)
        configured_repo = config.adapter.parameters.get("repo")
        if configured_repo:
            adapter.bind_repo(resolve_workspace_path(config, str(configured_repo)))
        return adapter

    async def run(self, config: ExperimentConfig, task: dict[str, Any]) -> ExperimentResult:
        run_id = str(uuid4())
        task_id = _task_id(task)
        started_at = datetime.now(UTC)
        started = perf_counter()
        run_dir = Path(config.output_dir).resolve() / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        adapter: TaskAdapter | None = None
        run_output: AdapterRunOutput | None = None
        evaluation: AdapterEvaluation | None = None
        error: Exception | None = None
        logger.info("experiment_started", extra={"run_id": run_id, "task_id": task_id})
        try:
            adapter = self._adapter(config)
            harness = self.harnesses.create(config.harness.type, config, run_id, task_id)
            adapter.setup(task, config)
            run_output = await adapter.run(task, harness, config)
            evaluation = adapter.evaluate(task, run_output, config)
        except Exception as caught:  # Runner boundary must convert failures into records.
            error = caught
            logger.exception("experiment_failed", extra={"run_id": run_id, "task_id": task_id})
        finally:
            if adapter is not None:
                try:
                    adapter.cleanup()
                except Exception as cleanup_error:
                    logger.exception("adapter_cleanup_failed", extra={"run_id": run_id})
                    if error is None:
                        error = cleanup_error

        elapsed_ms = (perf_counter() - started) * 1_000
        harness_result = run_output.harness_result if run_output else None
        trace = self._trace(harness_result.trace if harness_result else ())
        usage = self._usage(trace)
        retries = sum(1 for event in trace if event.type == "retry")
        failure = classify_failure(error=error, evaluation=evaluation, run_output=run_output)
        success = error is None and evaluation is not None and evaluation.success
        lineage = config.parameters.get("lineage", {})
        result = ExperimentResult(
            run_id=run_id,
            build_task_id=lineage.get("build_task_id"),
            build_result_id=lineage.get("build_result_id"),
            research_hypothesis_id=lineage.get("research_hypothesis_id"),
            experiment_plan_id=lineage.get("experiment_plan_id"),
            task_id=task_id,
            domain=config.experiment.domain,
            timestamp=started_at,
            completed_at=datetime.now(UTC),
            model=config.model.name,
            provider=config.model.provider,
            backend=config.model.backend,
            harness_type=config.harness.type,
            git_commit=_git_commit(Path.cwd()),
            config_version=config.config_version,
            config_fingerprint=config.fingerprint(),
            config=config.model_dump(mode="json"),
            success=success,
            quality=evaluation.quality if evaluation else None,
            latency_ms=elapsed_ms,
            input_tokens=evaluation.input_tokens
            if evaluation and evaluation.input_tokens is not None
            else usage[0],
            output_tokens=evaluation.output_tokens
            if evaluation and evaluation.output_tokens is not None
            else usage[1],
            cost_usd=evaluation.cost_usd
            if evaluation and evaluation.cost_usd is not None
            else float((run_output.metrics if run_output else {}).get("cost_usd") or 0),
            model_calls=harness_result.model_calls if harness_result else 0,
            tool_calls=harness_result.tool_calls if harness_result else 0,
            retries=retries,
            failure=failure,
            metrics=evaluation.metrics if evaluation else [],
            domain_metrics=evaluation.domain_metrics if evaluation else {},
            trace=trace,
            output=run_output.value if run_output else {},
            error=f"{type(error).__name__}: {error}" if error else None,
        )
        if adapter and run_output and evaluation:
            result.artifacts = self._write_artifacts(
                run_dir, adapter.artifacts(task, run_output, evaluation)
            )
        result.artifacts.extend(
            [
                Artifact(
                    name="trace.json",
                    path=str(run_dir / "trace.json"),
                    media_type="application/json",
                ),
                Artifact(
                    name="config.json",
                    path=str(run_dir / "config.json"),
                    media_type="application/json",
                ),
                Artifact(
                    name="evaluation.json",
                    path=str(run_dir / "evaluation.json"),
                    media_type="application/json",
                ),
            ]
        )
        if lineage.get("program_contents") is not None:
            program_path = run_dir / "program.md"
            program_path.write_text(str(lineage["program_contents"]), encoding="utf-8")
            decision_path = run_dir / "workflow_decision.json"
            decision_path.write_text(
                json.dumps(lineage.get("workflow_decision", {}), indent=2, default=str),
                encoding="utf-8",
            )
            result.artifacts.extend(
                [
                    Artifact(name="program.md", path=str(program_path), media_type="text/markdown"),
                    Artifact(
                        name="workflow_decision.json",
                        path=str(decision_path),
                        media_type="application/json",
                    ),
                ]
            )
        self._write_core(run_dir, result, config, evaluation)
        (self.tracker or build_tracker(config)).log(result, run_dir)
        (self.knowledge or ExperimentKnowledgeStore()).record(result)
        logger.info("experiment_completed", extra={"run_id": run_id, "success": success})
        return result

    async def run_batch(
        self, config: ExperimentConfig, tasks: list[dict[str, Any]] | None = None
    ) -> list[ExperimentResult]:
        if tasks is None:
            adapter = self._adapter(config)
            tasks = adapter.load_tasks(config)
        results: list[ExperimentResult] = []
        for task in tasks:
            results.append(await self.run(config, task))
        return results

    @staticmethod
    def _trace(events: tuple[Any, ...]) -> list[TraceEvent]:
        return [
            TraceEvent(
                sequence=event.sequence,
                type=event.type.value if hasattr(event.type, "value") else str(event.type),
                timestamp=event.timestamp,
                duration_ms=event.duration_ms,
                payload=event.payload,
            )
            for event in events
        ]

    @staticmethod
    def _usage(trace: list[TraceEvent]) -> tuple[int, int]:
        input_tokens = output_tokens = 0
        for event in trace:
            usage = event.payload.get("usage", {})
            input_tokens += int(usage.get("input_tokens", 0))
            output_tokens += int(usage.get("output_tokens", 0))
        return input_tokens, output_tokens

    @staticmethod
    def _write_artifacts(run_dir: Path, payloads: dict[str, Any]) -> list[Artifact]:
        directory = run_dir / "artifacts"
        directory.mkdir(parents=True, exist_ok=True)
        artifacts: list[Artifact] = []
        for name, payload in payloads.items():
            safe_name = Path(name).name
            path = directory / safe_name
            if isinstance(payload, (dict, list)):
                path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
                media_type = "application/json"
            elif isinstance(payload, bytes):
                path.write_bytes(payload)
                media_type = "application/octet-stream"
            else:
                path.write_text(str(payload), encoding="utf-8")
                media_type = "text/plain"
            artifacts.append(Artifact(name=safe_name, path=str(path), media_type=media_type))
        return artifacts

    @staticmethod
    def _write_core(
        run_dir: Path,
        result: ExperimentResult,
        config: ExperimentConfig,
        evaluation: AdapterEvaluation | None,
    ) -> None:
        (run_dir / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
        (run_dir / "config.json").write_text(config.model_dump_json(indent=2), encoding="utf-8")
        (run_dir / "trace.json").write_text(
            json.dumps([item.model_dump(mode="json") for item in result.trace], indent=2),
            encoding="utf-8",
        )
        (run_dir / "evaluation.json").write_text(
            evaluation and json.dumps(asdict(evaluation), indent=2, default=str) or "{}",
            encoding="utf-8",
        )
