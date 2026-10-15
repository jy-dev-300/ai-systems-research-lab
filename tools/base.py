from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel

from serving.models import ToolDefinition


class Tool(Protocol):
    name: str
    description: str
    arguments_model: type[BaseModel]

    def execute(self, arguments: BaseModel) -> Any:
        ...

    def definition(self) -> ToolDefinition:
        ...


class BaseTool:
    name: str
    description: str
    arguments_model: type[BaseModel]

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.arguments_model.model_json_schema(),
        )
