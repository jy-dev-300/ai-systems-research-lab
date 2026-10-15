from __future__ import annotations

import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

from harness.research.models import EvidenceType, ResearchDocument


def load_local_documents(path: str | Path) -> list[ResearchDocument]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Research fixture must be a JSON list")
    return [ResearchDocument.model_validate(item) for item in payload]


class ArxivMetadataSource:
    """Small, polite arXiv Atom connector; callers control query and result count."""

    endpoint = "https://export.arxiv.org/api/query"

    def search(self, query: str, *, max_results: int = 5) -> list[ResearchDocument]:
        if not 1 <= max_results <= 20:
            raise ValueError("max_results must be between 1 and 20")
        parameters = urllib.parse.urlencode(
            {"search_query": f"all:{query}", "start": 0, "max_results": max_results}
        )
        request = urllib.request.Request(
            f"{self.endpoint}?{parameters}",
            headers={"User-Agent": "optimo-research-harness/0.1 (metadata-only)"},
        )
        with urllib.request.urlopen(request, timeout=15) as response:  # noqa: S310
            root = ET.fromstring(response.read())
        namespace = {"atom": "http://www.w3.org/2005/Atom"}
        documents: list[ResearchDocument] = []
        for entry in root.findall("atom:entry", namespace):
            identifier = (entry.findtext("atom:id", default="", namespaces=namespace)).strip()
            published = entry.findtext("atom:published", default="", namespaces=namespace)
            documents.append(
                ResearchDocument(
                    id=identifier.rsplit("/", 1)[-1],
                    title=" ".join(
                        entry.findtext("atom:title", default="", namespaces=namespace).split()
                    ),
                    authors=[
                        author.findtext("atom:name", default="", namespaces=namespace)
                        for author in entry.findall("atom:author", namespace)
                    ],
                    publication_date=date.fromisoformat(published[:10]) if published else None,
                    venue="arXiv",
                    abstract=" ".join(
                        entry.findtext("atom:summary", default="", namespaces=namespace).split()
                    ),
                    url=identifier,
                    tags=["arxiv"],
                    provenance=f"arXiv API query: {query}",
                    evidence_type=EvidenceType.PAPER,
                )
            )
        return documents
