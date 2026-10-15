from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from backend.lab_dashboard import build_lab_overview
from harness.models import ExperimentResult, Metric


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, default=str), encoding="utf-8")


def test_lab_overview_aggregates_real_artifact_shapes(tmp_path: Path) -> None:
    result = ExperimentResult(
        run_id="run-bo-1",
        task_id="query-1",
        domain="Bo",
        model="deterministic-mock",
        provider="mock",
        backend="deterministic",
        harness_type="simple_loop",
        config_version="1.0",
        config_fingerprint="abc",
        config={"experiment": {"name": "bo-baseline"}},
        success=True,
        quality=1.0,
        latency_ms=5_000,
        metrics=[Metric(name="retrieval_latency_ms", value=4_500, unit="ms")],
        domain_metrics={"recall_at_k": 1.0},
    )
    _write(
        tmp_path / ".harness/runs/run-bo-1/result.json",
        result.model_dump(mode="json"),
    )
    _write(
        tmp_path / ".harness/programs/program-1/program_result.json",
        {
            "program_run_id": "program-1",
            "project": "Bo",
            "status": "review_required",
            "summary": "Candidate requires review.",
            "started_at": datetime.now(UTC).isoformat(),
            "outputs": {
                "candidate_experiments": [
                    {
                        "id": "candidate-1",
                        "candidate_technique": "smaller reranking pool",
                        "mechanism": "Reduce expensive reranker work.",
                        "expected_metric_affected": ["retrieval_latency_ms"],
                        "assumptions": ["Must be verified internally."],
                    }
                ]
            },
        },
    )
    _write(
        tmp_path / "research/fixtures/research_documents.json",
        [
            {
                "id": "paper-1",
                "title": "Efficient retrieval",
                "authors": ["Researcher"],
                "publication_date": "2026-01-01",
                "venue": "TestConf",
                "abstract": "A retrieval paper.",
                "url": "https://example.com/paper",
                "tags": ["retrieval"],
                "evidence_type": "paper",
            }
        ],
    )

    overview = build_lab_overview(tmp_path)

    assert overview.counts["experiments"] == 1
    assert overview.counts["candidates"] == 1
    assert overview.experiments[0].is_mock
    assert any(item.title == "Retrieval dominates query latency" for item in overview.opportunities)
    assert any(item.status == "candidate" for item in overview.opportunities)
    assert overview.research_documents[0].id == "paper-1"
    assert overview.activity[0].kind == "program"
