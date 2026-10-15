from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_demo_runs_all_deterministic_tasks() -> None:
    response = client.post("/api/runs/demo")
    assert response.status_code == 200
    body = response.json()
    assert body["backend"] == "deterministic-mock"
    assert len(body["tasks"]) == 3
    assert all(task["correct"] for task in body["tasks"])
    assert "not inference performance" in body["warning"]
