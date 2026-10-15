from __future__ import annotations

import math
import re
from collections import Counter

from harness.research.models import ResearchDocument


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_-]+", text.casefold())


class HybridResearchRetriever:
    """Dependency-light lexical plus hashed semantic retrieval."""

    def __init__(self, documents: list[ResearchDocument], dimensions: int = 128) -> None:
        self.documents = documents
        self.dimensions = dimensions

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in _tokens(text):
            vector[hash(token) % self.dimensions] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def search(self, query: str, *, limit: int = 5) -> list[tuple[ResearchDocument, float]]:
        query_terms = Counter(_tokens(query))
        query_vector = self._vector(query)
        scored: list[tuple[ResearchDocument, float]] = []
        for document in self.documents:
            text = " ".join([document.title, document.abstract, document.text, *document.tags])
            terms = Counter(_tokens(text))
            lexical = sum(min(count, terms[token]) for token, count in query_terms.items())
            semantic = sum(a * b for a, b in zip(query_vector, self._vector(text), strict=True))
            score = lexical + semantic
            if score > 0:
                scored.append((document, score))
        return sorted(scored, key=lambda item: (-item[1], item[0].id))[:limit]
