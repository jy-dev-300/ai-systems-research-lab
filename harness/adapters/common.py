from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.models import BenchmarkTask
from harness.models import ExperimentConfig
from harnesses.contracts import Harness, HarnessResult


def resolve_workspace_path(config: ExperimentConfig, configured: str) -> Path:
    path = Path(configured)
    if path.is_absolute():
        return path
    # Configs run from optimo; sibling repositories live one directory above.
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate.resolve()
    return (Path(__file__).resolve().parents[3] / path).resolve()


def load_json_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("tasks", [payload])
    if not isinstance(payload, list):
        raise ValueError(f"Dataset must be a list or JSONL file: {path}")
    return payload


async def run_reasoning_harness(
    harness: Harness,
    *,
    task_id: str,
    prompt: str,
    expected_answer: str,
) -> HarnessResult:
    task = BenchmarkTask(
        id=task_id,
        prompt=prompt,
        expected_answer=expected_answer,
        expected_tool_calls=[],
        tags=["domain-adapter"],
    )
    return await harness.run(task)
