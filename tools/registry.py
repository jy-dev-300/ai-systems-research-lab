from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import ValidationError

from serving.models import ToolDefinition
from tools.base import Tool


class UnknownToolError(ValueError):
    pass


class InvalidToolArgumentsError(ValueError):
    pass


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool]) -> None:
        self._tools = {tool.name: tool for tool in tools}
        if not self._tools:
            raise ValueError("ToolRegistry requires at least one tool")

    def definitions(self) -> list[ToolDefinition]:
        return [tool.definition() for tool in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            raise UnknownToolError(name)
        try:
            validated = tool.arguments_model.model_validate(arguments)
        except ValidationError as error:
            raise InvalidToolArgumentsError(str(error)) from error
        return tool.execute(validated)

    def names(self) -> set[str]:
        return set(self._tools)


def default_tool_registry() -> ToolRegistry:
    from tools.implementations import ArithmeticTool, InventoryTool

    return ToolRegistry([ArithmeticTool(), InventoryTool()])
