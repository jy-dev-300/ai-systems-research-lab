from __future__ import annotations

import re
from pathlib import Path

from harness.programs.models import ProgramDefinition, ProgramDocument


class ProgramRegistry:
    def __init__(self, lab_root: str | Path | None = None) -> None:
        self.lab_root = Path(lab_root or Path.cwd()).resolve()
        self._projects = {
            "bo": ProgramDefinition(
                project="Bo",
                adapter="bo",
                repo="../natural-language-file-search-rag",
                evaluation_config="configs/bo/hybrid.yaml",
                aliases=("bo",),
            ),
            "myvoice": ProgramDefinition(
                project="MyVoice",
                adapter="myvoice",
                repo="../myvoice",
                evaluation_config="configs/myvoice/retrieval.yaml",
                aliases=("myvoice", "my voice"),
            ),
            "swe": ProgramDefinition(
                project="SWE-bench",
                adapter="swebench",
                repo="../swe-bench",
                evaluation_config="configs/swebench/simple_loop.yaml",
                aliases=("swe", "swe-bench", "swebench"),
            ),
            "maniskill": ProgramDefinition(
                project="ManiSkill",
                adapter="maniskill",
                repo="../ManiSkill",
                evaluation_config="configs/maniskill/reach_smoke.yaml",
                aliases=("maniskill", "physical-ai", "physical ai"),
            ),
            "harness": ProgramDefinition(
                project="Harness",
                adapter="harness",
                repo=".",
                evaluation_config="configs/harness/baseline.yaml",
                aliases=("harness", "optimo"),
            ),
        }

    def resolve(self, program: ProgramDocument) -> ProgramDefinition:
        stem = Path(program.filename).stem.casefold()
        stem = stem.removesuffix("-program")
        candidates = [key for key in self._projects if stem == key or stem.startswith(f"{key}-")]
        heading = re.search(r"^#\s+(.+?)\s*$", program.contents, flags=re.MULTILINE)
        if not candidates and heading:
            title = heading.group(1).casefold()
            candidates = [
                key
                for key, value in self._projects.items()
                if title == key or title in value.aliases
            ]
        if len(candidates) != 1:
            raise ValueError(
                f"Cannot resolve one project for {program.filename!r}; use a registered project "
                "prefix or matching Markdown H1"
            )
        return self._projects[candidates[0]]

    def repo_path(self, definition: ProgramDefinition) -> Path:
        return (self.lab_root / definition.repo).resolve()

    def config_path(self, definition: ProgramDefinition) -> Path:
        return (self.lab_root / definition.evaluation_config).resolve()

    def definitions(self) -> tuple[ProgramDefinition, ...]:
        return tuple(self._projects.values())
