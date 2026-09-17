"""Read certification catalog and corpus quality without altering source JSON."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from certification_assistant.rag.pipeline import corpus_quality_report
from certification_assistant.schemas import RAGChunk
from pydantic import TypeAdapter

from ..core.config import get_settings
from ..schemas.api import CertificationSummary


@lru_cache(maxsize=1)
def load_chunks() -> list[RAGChunk]:
    path = get_settings().chunks_path
    if not path.exists():
        raise RuntimeError(f"RAG chunks are missing: {path}")
    return TypeAdapter(list[RAGChunk]).validate_json(path.read_text(encoding="utf-8"))


def _certification_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    certification = payload.get("certification", {})
    domains = payload.get("domains", []) if isinstance(payload.get("domains", []), list) else []
    topics = sum(
        len(domain.get("topics", []))
        for domain in domains
        if isinstance(domain, dict) and isinstance(domain.get("topics", []), list)
    )
    return {
        "certification_id": str(
            certification.get("id") or certification.get("certification_id") or path.stem
        ).strip(),
        "certification_name": str(
            certification.get("name") or certification.get("certification_name") or path.stem
        ).strip(),
        "provider": str(certification.get("provider") or "Unknown").strip(),
        "domains": len(domains),
        "topics": topics,
    }


@lru_cache(maxsize=1)
def list_certifications() -> list[CertificationSummary]:
    settings = get_settings()
    chunks = load_chunks()
    quality = corpus_quality_report(chunks)
    summaries: list[CertificationSummary] = []
    for path in sorted(settings.certifications_dir.glob("*.json")):
        base = _certification_payload(path)
        report = quality.get(
            base["certification_id"],
            {
                "chunks": 0,
                "detailed_chunks": 0,
                "question_generation_ready": False,
                "warning": "No usable RAG chunks were produced from this certification file.",
            },
        )
        summaries.append(
            CertificationSummary(
                **base,
                chunks=report["chunks"],
                detailed_chunks=report["detailed_chunks"],
                question_generation_ready=report["question_generation_ready"],
                warning=report["warning"],
            )
        )
    return summaries


def get_certification(certification_id: str) -> CertificationSummary | None:
    return next(
        (item for item in list_certifications() if item.certification_id == certification_id),
        None,
    )


def chunks_for_certification(certification_id: str) -> list[RAGChunk]:
    return [chunk for chunk in load_chunks() if chunk.certification_id == certification_id]

