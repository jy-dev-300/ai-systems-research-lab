from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from harness.models import ExperimentConfig, ExperimentResult


def _flatten(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(_flatten(item, name))
        elif isinstance(item, (str, int, float, bool)) or item is None:
            result[name] = item
        else:
            result[name] = json.dumps(item, sort_keys=True, default=str)
    return result


class ExperimentTracker(Protocol):
    def log(self, result: ExperimentResult, run_dir: Path) -> None: ...


class LocalTracker:
    def log(self, result: ExperimentResult, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")


class MLflowTracker:
    def __init__(self, config: ExperimentConfig) -> None:
        try:
            import mlflow
        except ImportError as error:
            raise RuntimeError(
                "MLflow tracking requested but mlflow is not installed; "
                "run: pip install -e '.[tracking]'"
            ) from error
        self.mlflow = mlflow
        self.uri = config.tracking.uri
        self.experiment = config.tracking.experiment or config.experiment.domain

    def log(self, result: ExperimentResult, run_dir: Path) -> None:
        mlflow = self.mlflow
        mlflow.set_tracking_uri(self.uri)
        mlflow.set_experiment(self.experiment)
        with mlflow.start_run(run_name=result.run_id, tags={"harness_run_id": result.run_id}):
            params = _flatten(result.config)
            params.update(
                {
                    "task": result.task_id,
                    "model": result.model,
                    "backend": result.backend,
                    "harness_architecture": result.harness_type,
                }
            )
            mlflow.log_params({key: str(value)[:500] for key, value in params.items()})
            mlflow.log_metrics(result.metric_values())
            tags = {
                "domain": result.domain,
                "git_commit": result.git_commit or "unavailable",
                "config_version": result.config_version,
                "failure_type": result.failure.type.value if result.failure else "none",
            }
            optional_lineage = {
                "build_task_id": result.build_task_id,
                "build_result_id": result.build_result_id,
                "research_hypothesis_id": result.research_hypothesis_id,
                "experiment_plan_id": result.experiment_plan_id,
            }
            tags.update(
                {name: value for name, value in optional_lineage.items() if value is not None}
            )
            program_lineage = result.config.get("parameters", {}).get("lineage", {})
            tags.update(
                {
                    name: str(program_lineage[name])[:500]
                    for name in (
                        "program_run_id",
                        "program_filename",
                        "program_sha256",
                        "workflow",
                        "workflow_classifier",
                        "workflow_classifier_version",
                        "workflow_prompt_version",
                    )
                    if program_lineage.get(name) is not None
                }
            )
            mlflow.set_tags(tags)
            for artifact in result.artifacts:
                path = Path(artifact.path)
                if path.exists():
                    mlflow.log_artifact(str(path), artifact_path="artifacts")
            for core in ("result.json", "trace.json", "config.json", "evaluation.json"):
                path = run_dir / core
                if path.exists():
                    mlflow.log_artifact(str(path), artifact_path="run")


def build_tracker(config: ExperimentConfig) -> ExperimentTracker:
    return MLflowTracker(config) if config.tracking.type == "mlflow" else LocalTracker()
