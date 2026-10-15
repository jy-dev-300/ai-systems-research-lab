from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from harness.models import ExperimentResult


class ExperimentKnowledgeStore:
    """Small index over run summaries; artifacts remain in MLflow/local run storage."""

    def __init__(self, path: str | Path = ".harness/knowledge.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY, domain TEXT NOT NULL, task_id TEXT NOT NULL,
                harness_type TEXT NOT NULL, success INTEGER NOT NULL, quality REAL,
                failure_type TEXT, failure_subtype TEXT, result_json TEXT NOT NULL,
                created_at TEXT NOT NULL)"""
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def record(self, result: ExperimentResult) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    result.run_id,
                    result.domain,
                    result.task_id,
                    result.harness_type,
                    int(result.success),
                    result.quality,
                    result.failure.type.value if result.failure else None,
                    result.failure.subtype if result.failure else None,
                    result.model_dump_json(),
                    result.timestamp.isoformat(),
                ),
            )

    def query(self, *, domain: str, task_id: str | None = None) -> list[ExperimentResult]:
        sql = "SELECT result_json FROM runs WHERE domain = ?"
        params: list[Any] = [domain]
        if task_id is not None:
            sql += " AND task_id = ?"
            params.append(task_id)
        sql += " ORDER BY created_at"
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [ExperimentResult.model_validate_json(row[0]) for row in rows]

    def summarize(self, *, domain: str, task_id: str | None = None) -> dict[str, Any]:
        results = self.query(domain=domain, task_id=task_id)
        by_harness: dict[str, list[ExperimentResult]] = {}
        for result in results:
            by_harness.setdefault(result.harness_type, []).append(result)
        return {
            "domain": domain,
            "task_id": task_id,
            "runs": len(results),
            "configurations": {
                name: {
                    "runs": len(items),
                    "success_rate": sum(item.success for item in items) / len(items),
                    "mean_quality": sum(item.quality or 0 for item in items) / len(items),
                    "failure_modes": sorted(
                        {
                            item.failure.subtype
                            for item in items
                            if item.failure and item.failure.subtype
                        }
                    ),
                }
                for name, items in by_harness.items()
            },
        }
