from __future__ import annotations

from typing import Protocol

from serving.models import ModelRequest, ModelResponse


class ServingBackend(Protocol):
    name: str

    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate one structured agent action."""
        ...
    async def health(self) -> bool:
        """Return whether this backend can currently serve requests."""
        ...
