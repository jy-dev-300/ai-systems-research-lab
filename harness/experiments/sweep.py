from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Any

import yaml

from harness.models import ExperimentConfig


def _assign(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    target = data
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        child = target.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"Sweep path crosses a non-mapping value: {dotted_key}")
        target = child
    target[parts[-1]] = value


def expand_sweep(payload: dict[str, Any]) -> list[ExperimentConfig]:
    base = payload.get("base")
    grid = payload.get("grid")
    if not isinstance(base, dict) or not isinstance(grid, dict) or not grid:
        raise ValueError("Sweep requires non-empty 'base' and 'grid' mappings")
    keys = list(grid)
    values: list[list[Any]] = []
    for key in keys:
        choices = grid[key]
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"Sweep values for {key} must be a non-empty list")
        values.append(choices)
    configs: list[ExperimentConfig] = []
    for combination in itertools.product(*values):
        candidate = copy.deepcopy(base)
        for key, value in zip(keys, combination, strict=True):
            _assign(candidate, key, value)
        configs.append(ExperimentConfig.model_validate(candidate))
    return configs


def load_sweep(path: str | Path) -> list[ExperimentConfig]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Sweep file must contain a mapping")
    return expand_sweep(payload)
