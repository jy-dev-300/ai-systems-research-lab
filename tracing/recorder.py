from __future__ import annotations

from typing import Any

from tracing.events import TraceEvent, TraceEventType


class TraceRecorder:
    """Create ordered, reconstructable, and sanitized trace events."""

    def __init__(self, run_id: str, task_id: str) -> None:
        self.run_id = run_id
        self.task_id = task_id
        self.events: list[TraceEvent] = []

    def record(
        self,
        event_type: TraceEventType,
        payload: dict[str, Any],
        duration_ms: float | None = None,
    ) -> TraceEvent:
        event = TraceEvent(
            run_id=self.run_id,
            task_id=self.task_id,
            sequence=len(self.events),
            type=event_type,
            duration_ms=duration_ms,
            payload=self._sanitize(payload),
        )
        self.events.append(event)
        return event

    @classmethod
    def _sanitize(cls, value: Any, depth: int = 0) -> Any:
        """Return a bounded, JSON-friendly copy with common secrets redacted."""
        if depth >= 8:
            return "<max-depth>"
        if isinstance(value, dict):
            sanitized: dict[str, Any] = {}
            for index, (key, item) in enumerate(value.items()):
                if index >= 100:
                    sanitized["<truncated>"] = f"{len(value) - 100} fields omitted"
                    break
                normalized = str(key).lower().replace("-", "_")
                if (
                    any(
                        marker in normalized
                        for marker in (
                            "password",
                            "secret",
                            "api_key",
                            "authorization",
                            "access_token",
                        )
                    )
                    or normalized == "token"
                    or normalized.endswith("_token")
                ):
                    sanitized[str(key)] = "<redacted>"
                else:
                    sanitized[str(key)] = cls._sanitize(item, depth + 1)
            return sanitized
        if isinstance(value, (list, tuple)):
            items = [cls._sanitize(item, depth + 1) for item in value[:100]]
            if len(value) > 100:
                items.append(f"<{len(value) - 100} items omitted>")
            return items
        if isinstance(value, str) and len(value) > 4_096:
            return f"{value[:4_096]}<truncated {len(value) - 4_096} chars>"
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if hasattr(value, "model_dump"):
            return cls._sanitize(value.model_dump(), depth + 1)
        return str(value)

    def snapshot(self) -> tuple[TraceEvent, ...]:
        return tuple(self.events)
