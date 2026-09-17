"""Helpers for converting retrieval results to a bounded prompt context."""

from __future__ import annotations

import json
from collections.abc import Sequence

from ..schemas import RetrievalResult


def build_context_payload(results: Sequence[RetrievalResult], max_chars: int = 6000) -> str:
    records = [
        {
            "chunk_id": result.chunk_id,
            "certification": result.certification,
            "domain": result.domain,
            "topic": result.topic,
            "subtopics": result.subtopics,
            "text": result.text[:max_chars],
            "source_url": result.source.source_url,
        }
        for result in results
    ]
    return json.dumps(records, ensure_ascii=False, indent=2)


__all__ = ["build_context_payload"]

