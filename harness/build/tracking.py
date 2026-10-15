from __future__ import annotations

from pathlib import Path
from typing import Protocol

from harness.build.models import AppliedChange, BuildResult


class BuildTracker(Protocol):
    def persist(self, result: BuildResult, changes: tuple[AppliedChange, ...]) -> BuildResult: ...


class LocalBuildTracker:
    """Stores the complete, reviewable build record and its exact applied diffs."""

    def persist(self, result: BuildResult, changes: tuple[AppliedChange, ...]) -> BuildResult:
        directory = Path(result.task.output_dir).resolve() / result.build_id
        directory.mkdir(parents=True, exist_ok=True)
        diff_path = directory / "changes.patch"
        diff_path.write_text("\n".join(change.diff for change in changes), encoding="utf-8")
        result.diff_artifact = str(diff_path)
        (directory / "build_result.json").write_text(
            result.model_dump_json(indent=2), encoding="utf-8"
        )
        (directory / "build_task.json").write_text(
            result.task.model_dump_json(indent=2), encoding="utf-8"
        )
        (directory / "trace.json").write_text(
            "[\n"
            + ",\n".join(event.model_dump_json(indent=2) for event in result.traces)
            + "\n]\n",
            encoding="utf-8",
        )
        return result
