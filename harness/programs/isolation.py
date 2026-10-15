from __future__ import annotations

import subprocess
from pathlib import Path


class WorktreeManager:
    """Creates unique detached worktrees without altering the source checkout."""

    def __init__(self, root: str | Path = ".harness/worktrees") -> None:
        self.root = Path(root).resolve()

    def create(self, repo: str | Path, run_id: str) -> tuple[Path, str | None, str]:
        source = Path(repo).resolve()
        commit = self._git(source, "rev-parse", "HEAD")
        if not commit:
            raise RuntimeError(f"Isolated modification requires a Git worktree: {source}")
        target = self.root / run_id
        if target.exists():
            raise FileExistsError(f"Worktree already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        process = subprocess.run(
            ["git", "worktree", "add", "--detach", str(target), commit],
            cwd=source,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if process.returncode:
            raise RuntimeError(process.stderr.strip() or "git worktree add failed")
        return target, None, commit

    @staticmethod
    def inspect(repo: str | Path) -> tuple[str | None, bool | None]:
        source = Path(repo).resolve()
        commit = WorktreeManager._git(source, "rev-parse", "HEAD")
        if not commit:
            return None, None
        status = WorktreeManager._git(source, "status", "--porcelain")
        return commit, bool(status)

    @staticmethod
    def _git(repo: Path, *args: str) -> str | None:
        try:
            process = subprocess.run(
                ["git", *args], cwd=repo, capture_output=True, text=True, timeout=10, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return process.stdout.strip() if process.returncode == 0 else None
