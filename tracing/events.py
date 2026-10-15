from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class TraceEventType(StrEnum):
    MODEL_CALL = "model_call"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    RETRY = "retry"
    STATE_UPDATE = "state_update"
    ERROR = "error"
    TERMINATION = "termination"


class TraceEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    task_id: str
    sequence: int
    type: TraceEventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    duration_ms: float | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
