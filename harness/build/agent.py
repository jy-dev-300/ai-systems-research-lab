from __future__ import annotations

import difflib
import hashlib
from pathlib import Path

from harness.build.backend import CodingBackend
from harness.build.models import (
    AppliedChange,
    BuildDiagnosis,
    BuildPlan,
    BuildProposal,
    BuildTask,
    BuildValidation,
    FileChange,
    RepositoryContext,
)


class BuildAgent:
    """One general build agent backed by a pluggable change proposer."""

    def __init__(self, backend: CodingBackend) -> None:
        self.backend = backend

    async def plan(self, task: BuildTask, context: RepositoryContext) -> BuildPlan:
        return await self.backend.plan(task, context)

    async def propose_changes(
        self,
        task: BuildTask,
        context: RepositoryContext,
        plan: BuildPlan,
        iteration: int,
        diagnosis: BuildDiagnosis | None,
    ) -> BuildProposal:
        return await self.backend.propose_changes(task, context, plan, iteration, diagnosis)

    async def diagnose(
        self,
        task: BuildTask,
        plan: BuildPlan,
        iteration: int,
        validation: BuildValidation,
    ) -> BuildDiagnosis:
        return await self.backend.diagnose(task, plan, iteration, validation)

    def apply(self, task: BuildTask, proposal: BuildProposal) -> tuple[AppliedChange, ...]:
        root = Path(task.target_project).resolve()
        prepared: list[tuple[FileChange, Path, str | None]] = []
        seen_paths: set[Path] = set()
        # Validate the complete proposal before writing any file, so an invalid
        # later change cannot leave a partially applied iteration.
        for change in proposal.changes:
            path = self._contained_path(root, change.path)
            if path in seen_paths:
                raise ValueError(f"Build proposal contains duplicate path: {change.path}")
            seen_paths.add(path)
            before = path.read_text(encoding="utf-8") if path.exists() else None
            if change.operation == "delete":
                if "delete" not in task.allowed_actions or (
                    "delete" not in task.approved_high_risk_actions
                ):
                    raise PermissionError("Delete operation was not explicitly approved")
                if before is None:
                    raise FileNotFoundError(f"Delete target does not exist: {change.path}")
            elif change.operation == "create" and before is not None:
                raise FileExistsError(f"Create change would overwrite existing file: {change.path}")
            elif change.operation == "update" and before is None:
                raise FileNotFoundError(f"Update target does not exist: {change.path}")
            prepared.append((change, path, before))

        applied: list[AppliedChange] = []
        for change, path, before in prepared:
            if change.operation == "delete":
                assert before is not None
                path.unlink()
                applied.append(
                    AppliedChange(
                        path=path.relative_to(root).as_posix(),
                        operation=change.operation,
                        before_sha256=self._sha(before),
                        after_sha256=None,
                        diff="".join(
                            difflib.unified_diff(
                                before.splitlines(keepends=True),
                                [],
                                fromfile=f"a/{change.path}",
                                tofile="/dev/null",
                            )
                        ),
                    )
                )
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            after = change.content or ""
            path.write_text(after, encoding="utf-8")
            applied.append(
                AppliedChange(
                    path=path.relative_to(root).as_posix(),
                    operation=change.operation,
                    before_sha256=self._sha(before) if before is not None else None,
                    after_sha256=self._sha(after),
                    diff="".join(
                        difflib.unified_diff(
                            (before or "").splitlines(keepends=True),
                            after.splitlines(keepends=True),
                            fromfile=f"a/{change.path}",
                            tofile=f"b/{change.path}",
                        )
                    ),
                )
            )
        return tuple(applied)

    @staticmethod
    def _contained_path(root: Path, relative: str) -> Path:
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError(f"Build change escapes target project: {relative}") from error
        return candidate

    @staticmethod
    def _sha(value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()
