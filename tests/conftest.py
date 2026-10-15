import pytest

from benchmarks.deterministic import task_by_id
from serving.mock import DeterministicMockBackend
from tools.registry import default_tool_registry


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-student",
        action="store_true",
        default=False,
        help="run specification tests for intentionally incomplete student assignments",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-student"):
        return
    skip_student = pytest.mark.skip(reason="pass --run-student when implementing this assignment")
    for item in items:
        if "student" in item.keywords:
            item.add_marker(skip_student)


@pytest.fixture
def mock_backend() -> DeterministicMockBackend:
    return DeterministicMockBackend()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def tool_registry():
    return default_tool_registry()


@pytest.fixture
def addition_task():
    return task_by_id("add-017")
