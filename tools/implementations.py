from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from tools.base import BaseTool


class ArithmeticArguments(BaseModel):
    operation: Literal["add", "multiply"]
    left: float
    right: float


class ArithmeticTool(BaseTool):
    name = "arithmetic"
    description = "Add or multiply two numbers. Use this instead of mental arithmetic."
    arguments_model = ArithmeticArguments

    def execute(self, arguments: ArithmeticArguments) -> float:
        if arguments.operation == "add":
            return arguments.left + arguments.right
        return arguments.left * arguments.right


class InventoryArguments(BaseModel):
    sku: str


class InventoryTool(BaseTool):
    name = "inventory_lookup"
    description = "Return the deterministic stock count for an exact SKU."
    arguments_model = InventoryArguments
    inventory = {"KEY-104": 7, "MOUSE-220": 0, "USB-032": 18}

    def execute(self, arguments: InventoryArguments) -> int:
        if arguments.sku not in self.inventory:
            raise KeyError(f"Unknown SKU: {arguments.sku}")
        return self.inventory[arguments.sku]
