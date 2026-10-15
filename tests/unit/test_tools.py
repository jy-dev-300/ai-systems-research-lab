import pytest

from tools.registry import InvalidToolArgumentsError, UnknownToolError, default_tool_registry


def test_registry_executes_validated_tool() -> None:
    registry = default_tool_registry()
    result = registry.execute("arithmetic", {"operation": "add", "left": 17, "right": 25})
    assert result == 42


def test_registry_rejects_invalid_arguments() -> None:
    registry = default_tool_registry()
    with pytest.raises(InvalidToolArgumentsError):
        registry.execute("arithmetic", {"operation": "divide", "left": 6, "right": 2})


def test_registry_rejects_unknown_tool() -> None:
    registry = default_tool_registry()
    with pytest.raises(UnknownToolError):
        registry.execute("shell", {"command": "anything"})
