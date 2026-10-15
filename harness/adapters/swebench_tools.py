from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TestExecution:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str


class RepositoryWorkspace:
    """Path-contained coding operations for a checked-out SWE-bench task repository."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Task repository not found: {self.root}")

    def _path(self, relative: str) -> Path:
        candidate = (self.root / relative).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as error:
            raise ValueError(f"Path escapes task repository: {relative}") from error
        return candidate

    def list_files(self, pattern: str = "*") -> list[str]:
        return sorted(
            path.relative_to(self.root).as_posix()
            for path in self.root.rglob(pattern)
            if path.is_file() and ".git" not in path.parts
        )

    def search(self, needle: str, pattern: str = "*.py") -> list[dict[str, str | int]]:
        matches: list[dict[str, str | int]] = []
        for relative in self.list_files(pattern):
            try:
                lines = self._path(relative).read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(lines, 1):
                if needle in line:
                    matches.append({"path": relative, "line": line_number, "text": line})
        return matches

    def read_file(self, relative: str) -> str:
        return self._path(relative).read_text(encoding="utf-8")

    def write_file(self, relative: str, content: str) -> None:
        path = self._path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def run_tests(self, command: list[str], *, timeout_seconds: float = 300) -> TestExecution:
        if not command:
            raise ValueError("Test command cannot be empty")
        process = subprocess.run(
            command,
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        return TestExecution(command, process.returncode, process.stdout, process.stderr)
