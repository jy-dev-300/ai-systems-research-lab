from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ExpectedToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]


class BenchmarkTask(BaseModel):
    id: str
    prompt: str
    expected_answer: str
    expected_tool_calls: list[ExpectedToolCall]
    tags: list[str] = Field(default_factory=list)


class BenchmarkSuite(BaseModel):
    name: str
    version: str
    tasks: list[BenchmarkTask]
