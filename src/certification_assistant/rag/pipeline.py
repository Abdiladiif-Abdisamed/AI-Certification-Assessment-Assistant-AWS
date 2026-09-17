"""Topic-aware ingestion, chunking, OpenAI embeddings, and filtered FAISS retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import textwrap
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import numpy as np
from dotenv import load_dotenv

from ..schemas import RAGChunk, RAGDocument, RAGMetadata, RetrievalResult, SourceReference


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE = re.compile(r"[ \t]+")
BLANK_LINES = re.compile(r"\n{3,}")


def normalize_text(value: Any) -> str:
    """Normalize layout and malformed control characters without rewriting meaning."""

    if value is None:
        return ""
    text = str(value).replace("\ufeff", "").replace("\ufffd", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_CHARACTERS.sub("", text)
    text = "\n".join(WHITESPACE.sub(" ", line).strip() for line in text.splitlines())
    return BLANK_LINES.sub("\n\n", text).strip()


def _first_text(mapping: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = normalize_text(mapping.get(key))
        if value:
            return value
    return ""


def _value_to_text(value: Any, prefix: str = "") -> str:
    """Render supplied structured content while retaining its original facts and labels."""

    if value is None:
        return ""
    if isinstance(value, str) or isinstance(value, (int, float, bool)):
        scalar = normalize_text(value)
        return f"{prefix}: {scalar}" if prefix and scalar else scalar
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parts = [_value_to_text(item) for item in value]
        joined = "\n".join(part for part in parts if part)
        return f"{prefix}:\n{joined}" if prefix and joined else joined
    if isinstance(value, Mapping):
        parts = []
        for key, item in value.items():
            text = _value_to_text(item, normalize_text(key).replace("_", " "))
            if text:
                parts.append(text)
        joined = "\n".join(parts)
        return f"{prefix}:\n{joined}" if prefix and joined else joined
    return normalize_text(value)


def _normalize_source(source: Any) -> dict[str, str] | None:
    if isinstance(source, str):
        url = normalize_text(source)
        return (
            {"source_url": url, "source_title": "", "source_type": "official"}
            if url
            else None
        )
    if isinstance(source, Mapping):
        url = _first_text(source, "source_url", "url", "href")
        if not url:
            return None
        return {
            "source_url": url,
            "source_title": _first_text(source, "source_title", "title", "name"),
            "source_type": _first_text(source, "source_type", "type") or "official",
        }
    return None


def normalize_sources(raw_sources: Any) -> tuple[list[dict[str, str]], int]:
    sources = raw_sources if isinstance(raw_sources, list) else [raw_sources]
    normalized: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    duplicate_count = 0
    for raw_source in sources:
        source = _normalize_source(raw_source)
        if source is None:
            continue
        key = (
            source["source_url"].casefold(),
            source["source_title"].casefold(),
            source["source_type"].casefold(),
        )
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        normalized.append(source)
    return normalized, duplicate_count


def _topic_parts(topic_value: Any) -> tuple[str, list[str], str, bool, list[dict[str, str]]]:
    """Return topic, subtopics, content, has-detailed-content, and topic sources."""

    if isinstance(topic_value, str):
        topic = normalize_text(topic_value)
        return topic, [], topic, False, []
    if not isinstance(topic_value, Mapping):
        topic = normalize_text(topic_value)
        return topic, [], topic, False, []

    topic = _first_text(topic_value, "name", "topic", "title")
    subtopic_values = topic_value.get("subtopics", [])
    if not isinstance(subtopic_values, list):
        subtopic_values = [subtopic_values]
    subtopics: list[str] = []
    subtopic_details: list[str] = []
    for subtopic in subtopic_values:
        if isinstance(subtopic, Mapping):
            name = _first_text(subtopic, "name", "subtopic", "title")
            if name:
                subtopics.append(name)
            detail_values = [
                subtopic.get(key)
                for key in ("content", "description", "details", "text", "knowledge")
                if subtopic.get(key) not in (None, "", [], {})
            ]
            for detail in detail_values:
                rendered = _value_to_text(detail)
                if rendered:
                    subtopic_details.append(
                        f"{name}: {rendered}" if name and rendered != name else rendered
                    )
        else:
            name = normalize_text(subtopic)
            if name:
                subtopics.append(name)

    explicit_content_parts: list[str] = []
    for key in ("content", "description", "details", "text", "knowledge"):
        value = topic_value.get(key)
        if value not in (None, "", [], {}):
            rendered = _value_to_text(value)
            if rendered:
                explicit_content_parts.append(rendered)
    explicit_content_parts.extend(subtopic_details)
    has_details = bool(explicit_content_parts)
    if has_details:
        content = "\n".join(explicit_content_parts)
    else:
        content = "\n".join(part for part in [topic, *subtopics] if part)

    topic_sources, _ = normalize_sources(
        topic_value.get("official_sources")
        or topic_value.get("sources")
        or topic_value.get("source")
        or []
    )
    return topic, list(dict.fromkeys(subtopics)), content, has_details, topic_sources


def _document_text(metadata: RAGMetadata) -> str:
    lines = [
        f"Certification: {metadata.certification}",
        f"Provider: {metadata.provider}",
        f"Domain: {metadata.domain}",
    ]
    if metadata.domain_weight:
        lines.append(f"Domain weight: {metadata.domain_weight}")
    lines.append(f"Topic: {metadata.topic}")
    if metadata.subtopics:
        lines.append(f"Subtopics: {', '.join(metadata.subtopics)}")
    lines.extend(["Content:", metadata.content])
    return normalize_text("\n".join(lines))


def load_certification_documents(
    certifications_dir: str | Path,
) -> tuple[list[RAGDocument], dict[str, Any]]:
    """Dynamically load every JSON certification and flatten its topic hierarchy."""

    directory = Path(certifications_dir)
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No certification JSON files found in {directory}")

    documents: list[RAGDocument] = []
    certification_ids: list[str] = []
    total_domains = 0
    total_topics = 0
    documents_before_cleaning = 0
    duplicate_source_records = 0
    per_certification: dict[str, dict[str, Any]] = {}

    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot parse certification file {path}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"Certification file must contain a JSON object: {path}")

        certification = payload.get("certification", {})
        if not isinstance(certification, Mapping):
            raise ValueError(f"'certification' must be an object in {path}")
        certification_id = _first_text(certification, "id", "certification_id")
        certification_name = _first_text(certification, "name", "certification_name")
        provider = _first_text(certification, "provider")
        if not certification_id or not certification_name or not provider:
            raise ValueError(
                f"Missing certification id, name, or provider in source file {path}"
            )
        if certification_id in certification_ids:
            raise ValueError(f"Duplicate certification id: {certification_id}")
        certification_ids.append(certification_id)

        certification_sources, source_duplicates = normalize_sources(
            payload.get("official_sources", [])
        )
        duplicate_source_records += source_duplicates
        domains = payload.get("domains", [])
        if not isinstance(domains, list):
            raise ValueError(f"'domains' must be a list in {path}")
        total_domains += len(domains)
        certification_topic_count = 0
        certification_document_count = 0

        for domain_value in domains:
            if not isinstance(domain_value, Mapping):
                continue
            domain = _first_text(domain_value, "name", "domain", "title")
            domain_weight = _first_text(domain_value, "weight")
            domain_sources, domain_source_duplicates = normalize_sources(
                domain_value.get("official_sources")
                or domain_value.get("sources")
                or domain_value.get("source")
                or []
            )
            duplicate_source_records += domain_source_duplicates
            topics = domain_value.get("topics", [])
            if not isinstance(topics, list):
                topics = [topics]

            # Some expanded files may put content directly on a domain.
            if not topics and any(
                domain_value.get(key) not in (None, "", [], {})
                for key in ("content", "description", "details", "text", "knowledge")
            ):
                topics = [
                    {
                        "name": domain,
                        "content": next(
                            domain_value[key]
                            for key in ("content", "description", "details", "text", "knowledge")
                            if domain_value.get(key) not in (None, "", [], {})
                        ),
                    }
                ]

            total_topics += len(topics)
            certification_topic_count += len(topics)
            for topic_value in topics:
                documents_before_cleaning += 1
                topic, subtopics, content, has_details, topic_sources = _topic_parts(
                    topic_value
                )
                sources = topic_sources or domain_sources or certification_sources
                source = sources[0] if sources else {
                    "source_url": "",
                    "source_title": "",
                    "source_type": "official",
                }
                source_urls = list(
                    dict.fromkeys(item["source_url"] for item in sources if item["source_url"])
                )
                metadata = RAGMetadata(
                    certification_id=certification_id,
                    certification_name=certification_name,
                    certification=certification_name,
                    provider=provider,
                    domain=domain,
                    domain_weight=domain_weight,
                    topic=topic,
                    subtopics=subtopics,
                    content=normalize_text(content),
                    source_title=source["source_title"],
                    source_url=source["source_url"],
                    source_type=source["source_type"],
                    source_urls=source_urls,
                    content_detail_level="detailed" if has_details else "topic_label",
                )
                documents.append(RAGDocument(text=_document_text(metadata), metadata=metadata))
                certification_document_count += 1

        per_certification[certification_id] = {
            "name": certification_name,
            "file": path.name,
            "domains": len(domains),
            "topics": certification_topic_count,
            "documents_before_cleaning": certification_document_count,
            "note": normalize_text(payload.get("note")),
        }

    return documents, {
        "number_of_certifications": len(certification_ids),
        "number_of_domains": total_domains,
        "number_of_topics": total_topics,
        "documents_before_cleaning": documents_before_cleaning,
        "duplicate_source_records": duplicate_source_records,
        "certification_ids": certification_ids,
        "per_certification": per_certification,
    }


def clean_documents(
    documents: Iterable[RAGDocument],
) -> tuple[list[RAGDocument], dict[str, int]]:
    """Remove unusable and exact duplicate records after conservative normalization."""

    cleaned: list[RAGDocument] = []
    seen: set[str] = set()
    empty_removed = 0
    duplicate_removed = 0
    for document in documents:
        metadata = document.metadata.model_copy(
            update={
                "content": normalize_text(document.metadata.content),
                "topic": normalize_text(document.metadata.topic),
                "domain": normalize_text(document.metadata.domain),
                "subtopics": list(
                    dict.fromkeys(
                        normalize_text(value)
                        for value in document.metadata.subtopics
                        if normalize_text(value)
                    )
                ),
            }
        )
        if not metadata.content or not metadata.topic or not metadata.domain:
            empty_removed += 1
            continue
        normalized_document = RAGDocument(
            text=_document_text(metadata), metadata=metadata
        )
        fingerprint_payload = "\x1f".join(
            [
                metadata.certification_id.casefold(),
                metadata.domain.casefold(),
                metadata.topic.casefold(),
                metadata.content.casefold(),
                metadata.source_url.casefold(),
            ]
        )
        fingerprint = hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()
        if fingerprint in seen:
            duplicate_removed += 1
            continue
        seen.add(fingerprint)
        cleaned.append(normalized_document)
    return cleaned, {
        "documents_after_cleaning": len(cleaned),
        "empty_documents_removed": empty_removed,
        "duplicate_documents_removed": duplicate_removed,
    }


def _split_long_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if len(paragraphs) == 1:
        paragraphs = [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
            if part.strip()
        ]
    if len(paragraphs) == 1:
        words = text.split()
        paragraphs = []
        current: list[str] = []
        for word in words:
            if current and len(" ".join([*current, word])) > max_chars:
                paragraphs.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            paragraphs.append(" ".join(current))

    bounded_paragraphs: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            bounded_paragraphs.append(paragraph)
            continue
        bounded_paragraphs.extend(
            textwrap.wrap(
                paragraph,
                width=max_chars,
                break_long_words=True,
                break_on_hyphens=False,
                replace_whitespace=False,
                drop_whitespace=True,
            )
        )
    paragraphs = bounded_paragraphs

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if current and len(candidate) > max_chars:
            chunks.append(current)
            overlap_budget = min(
                overlap_chars, max(0, max_chars - len(paragraph) - 2)
            )
            overlap = current[-overlap_budget:] if overlap_budget else ""
            if overlap and " " in overlap:
                overlap = overlap.split(" ", 1)[1]
            current = f"{overlap}\n\n{paragraph}".strip()
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def chunk_documents(
    documents: Iterable[RAGDocument],
    *,
    max_chars: int = 3500,
    overlap_chars: int = 250,
) -> list[RAGChunk]:
    """Keep topic units intact and split only unusually long topic content."""

    if max_chars < 500:
        raise ValueError("max_chars must be at least 500 for meaningful semantic chunks")
    if overlap_chars < 0 or overlap_chars >= max_chars // 2:
        raise ValueError("overlap_chars must be non-negative and less than half max_chars")

    chunks: list[RAGChunk] = []
    for document in documents:
        metadata = document.metadata
        header_lines = [
            f"Certification: {metadata.certification}",
            f"Provider: {metadata.provider}",
            f"Domain: {metadata.domain}",
        ]
        if metadata.domain_weight:
            header_lines.append(f"Domain weight: {metadata.domain_weight}")
        header_lines.append(f"Topic: {metadata.topic}")
        if metadata.subtopics:
            header_lines.append(f"Subtopics: {', '.join(metadata.subtopics)}")
        header = "\n".join(header_lines)
        available_content_chars = max(500, max_chars - len(header) - len("\nContent:\n"))
        pieces = _split_long_text(
            metadata.content, available_content_chars, overlap_chars
        )
        for piece_index, piece in enumerate(pieces):
            chunk_text = normalize_text(f"{header}\nContent:\n{piece}")
            digest_input = "\x1f".join(
                [
                    metadata.certification_id,
                    metadata.domain,
                    metadata.topic,
                    metadata.source_url,
                    str(piece_index),
                    piece,
                ]
            )
            chunk_id = "chunk_" + hashlib.sha256(
                digest_input.encode("utf-8")
            ).hexdigest()[:20]
            chunks.append(
                RAGChunk(
                    chunk_id=chunk_id,
                    certification_id=metadata.certification_id,
                    certification=metadata.certification,
                    provider=metadata.provider,
                    domain=metadata.domain,
                    domain_weight=metadata.domain_weight,
                    topic=metadata.topic,
                    subtopics=metadata.subtopics,
                    text=chunk_text,
                    source_url=metadata.source_url,
                    source_title=metadata.source_title,
                    source_type=metadata.source_type,
                    source_urls=metadata.source_urls,
                    content_detail_level=metadata.content_detail_level,
                )
            )
    return chunks


def save_chunks(chunks: Sequence[RAGChunk], output_path: str | Path) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            [chunk.model_dump(mode="json") for chunk in chunks],
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def prepare_rag_data(
    certifications_dir: str | Path = "certifications",
    chunks_output: str | Path = "data/rag_chunks.json",
    *,
    max_chars: int = 3500,
    overlap_chars: int = 250,
) -> tuple[list[RAGChunk], dict[str, Any]]:
    documents, load_stats = load_certification_documents(certifications_dir)
    cleaned, clean_stats = clean_documents(documents)
    chunks = chunk_documents(
        cleaned, max_chars=max_chars, overlap_chars=overlap_chars
    )
    save_chunks(chunks, chunks_output)
    detail_counts = Counter(chunk.content_detail_level for chunk in chunks)
    stats = {
        **load_stats,
        **clean_stats,
        "total_chunks": len(chunks),
        "detailed_chunks": detail_counts.get("detailed", 0),
        "topic_label_chunks": detail_counts.get("topic_label", 0),
    }
    return chunks, stats


class EmbeddingProvider(Protocol):
    model_name: str

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class OpenAIEmbeddingProvider:
    """OpenAI embedding adapter with batching and dependency injection support."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        batch_size: int = 128,
        client: Any | None = None,
    ) -> None:
        load_dotenv()
        self.model_name = model or os.getenv(
            "OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        )
        self.batch_size = batch_size
        if client is not None:
            self.client = client
            return
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required to generate or query OpenAI embeddings"
            )
        from openai import OpenAI

        self.client = OpenAI(api_key=resolved_key)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed many texts via the OpenAI embeddings API, in batches.

        Called when building the vector index. Sends up to ``batch_size`` texts per
        request, re-sorts each response by index (the API may reorder), and returns a
        float32 matrix (one row per input) ready to add to FAISS.
        """
        if not texts:
            raise ValueError("Cannot embed an empty text collection")
        if any(not normalize_text(text) for text in texts):
            raise ValueError("Embedding input cannot contain an empty string")
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            response = self.client.embeddings.create(
                model=self.model_name, input=batch, encoding_format="float"
            )
            vectors.extend(
                item.embedding for item in sorted(response.data, key=lambda item: item.index)
            )
        return np.asarray(vectors, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single search query via the OpenAI embeddings API and return its
        float32 vector — used at retrieval time to find the most relevant chunks to
        ground question generation."""
        query = normalize_text(text)
        if not query:
            raise ValueError("query cannot be empty")
        response = self.client.embeddings.create(
            model=self.model_name, input=query, encoding_format="float"
        )
        return np.asarray(response.data[0].embedding, dtype="float32")


def _faiss_module() -> Any:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError(
            "FAISS is not installed. Install dependencies from requirements.txt."
        ) from exc
    return faiss


def _safe_index_name(certification_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", certification_id).strip("._")
    if not safe:
        raise ValueError(f"Cannot create index filename for {certification_id!r}")
    return safe


def build_vector_store(
    chunks: Sequence[RAGChunk | Mapping[str, Any]],
    output_dir: str | Path = "vector_store",
    *,
    embedding_provider: EmbeddingProvider | None = None,
) -> dict[str, Any]:
    """Embed chunks once and persist global plus certification-scoped FAISS indexes."""

    parsed = [
        chunk if isinstance(chunk, RAGChunk) else RAGChunk.model_validate(chunk)
        for chunk in chunks
    ]
    if not parsed:
        raise ValueError("Cannot build a vector store without chunks")
    provider = embedding_provider or OpenAIEmbeddingProvider()
    vectors = provider.embed_documents([chunk.text for chunk in parsed])
    if vectors.ndim != 2 or vectors.shape[0] != len(parsed) or vectors.shape[1] == 0:
        raise ValueError("Embedding provider returned an invalid matrix")

    faiss = _faiss_module()
    vectors = np.ascontiguousarray(vectors, dtype="float32")
    faiss.normalize_L2(vectors)
    destination = Path(output_dir)
    certification_directory = destination / "by_certification"
    certification_directory.mkdir(parents=True, exist_ok=True)

    global_index = faiss.IndexFlatIP(vectors.shape[1])
    global_index.add(vectors)
    faiss.write_index(global_index, str(destination / "index.faiss"))

    certification_entries: dict[str, dict[str, Any]] = {}
    for certification_id in sorted({chunk.certification_id for chunk in parsed}):
        positions = [
            index
            for index, chunk in enumerate(parsed)
            if chunk.certification_id == certification_id
        ]
        scoped_vectors = np.ascontiguousarray(vectors[positions], dtype="float32")
        scoped_index = faiss.IndexFlatIP(vectors.shape[1])
        scoped_index.add(scoped_vectors)
        filename = f"{_safe_index_name(certification_id)}.faiss"
        relative_path = Path("by_certification") / filename
        faiss.write_index(scoped_index, str(destination / relative_path))
        certification_entries[certification_id] = {
            "index_file": relative_path.as_posix(),
            "chunk_positions": positions,
            "chunk_count": len(positions),
        }

    manifest = {
        "format_version": 1,
        "status": "ready",
        "embedding_provider": "openai"
        if isinstance(provider, OpenAIEmbeddingProvider)
        else "injected",
        "embedding_model": provider.model_name,
        "dimension": int(vectors.shape[1]),
        "metric": "cosine_similarity_via_normalized_inner_product",
        "chunk_count": len(parsed),
        "certifications": certification_entries,
        "chunks": [chunk.model_dump(mode="json") for chunk in parsed],
    }
    (destination / "metadata.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def initialize_unbuilt_vector_store(
    chunks: Sequence[RAGChunk | Mapping[str, Any]],
    output_dir: str | Path = "vector_store",
    *,
    reason: str = "OPENAI_API_KEY is not configured; run the embedding step after configuration.",
) -> dict[str, Any]:
    """Create explicit, non-queryable placeholders without pretending vectors were generated."""

    parsed = [
        chunk if isinstance(chunk, RAGChunk) else RAGChunk.model_validate(chunk)
        for chunk in chunks
    ]
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    faiss = _faiss_module()
    faiss.write_index(faiss.IndexFlatIP(1), str(destination / "index.faiss"))
    manifest = {
        "format_version": 1,
        "status": "not_built",
        "reason": reason,
        "embedding_provider": "openai",
        "embedding_model": os.getenv(
            "OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        ),
        "dimension": None,
        "metric": "cosine_similarity_via_normalized_inner_product",
        "chunk_count": len(parsed),
        "certifications": {},
        "chunks": [chunk.model_dump(mode="json") for chunk in parsed],
    }
    (destination / "metadata.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


class FaissRetriever:
    """Retriever that selects a certification-specific index before vector search."""

    def __init__(
        self,
        vector_store_dir: str | Path = "vector_store",
        *,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        """Open a prebuilt vector store: load its manifest, verify it's 'ready', wire
        up the embedding provider (its model MUST match the one used to build the index),
        and load the chunk metadata. Per-certification indexes are read lazily on first use."""
        self.directory = Path(vector_store_dir)
        metadata_path = self.directory / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Missing {metadata_path}; build the vector store before retrieval"
            )
        self.manifest = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.manifest.get("status") != "ready":
            raise RuntimeError(
                f"Vector store is not ready: {self.manifest.get('reason', 'unknown reason')}"
            )
        self.provider = embedding_provider or OpenAIEmbeddingProvider(
            model=self.manifest["embedding_model"]
        )
        if self.provider.model_name != self.manifest["embedding_model"]:
            raise ValueError(
                "Query embedding model must match the model used to build the index"
            )
        self._faiss = _faiss_module()
        self._chunks = [RAGChunk.model_validate(item) for item in self.manifest["chunks"]]
        self._index_cache: dict[str, Any] = {}

    @property
    def certification_ids(self) -> list[str]:
        """Certification IDs that have a built, searchable index."""
        return sorted(self.manifest["certifications"])

    def _scoped_index(self, certification_id: str) -> tuple[Any, list[int]]:
        """Load (and cache) the FAISS index for ONE certification, plus the map from
        that index's local positions back to global chunk positions. Scoping to a
        single certification guarantees retrieval never mixes exams."""
        entry = self.manifest["certifications"].get(certification_id)
        if entry is None:
            raise ValueError(
                f"Unknown certification_id {certification_id!r}; available: "
                + ", ".join(self.certification_ids)
            )
        if certification_id not in self._index_cache:
            path = self.directory / entry["index_file"]
            if not path.exists():
                raise FileNotFoundError(f"Missing certification-scoped index: {path}")
            self._index_cache[certification_id] = self._faiss.read_index(str(path))
        return self._index_cache[certification_id], entry["chunk_positions"]

    def retrieve_context(
        self, query: str, certification_id: str, top_k: int = 5
    ) -> list[RetrievalResult]:
        """Semantic search for the ``top_k`` most relevant chunks of ONE certification.

        This is the RAG bridge into question generation: it embeds the query (OpenAI),
        cosine-searches that certification's scoped FAISS index, and returns ranked
        ``RetrievalResult`` objects (text + domain/topic + source), each carrying the
        provenance the generator must cite. An integrity check guarantees no chunk from
        another certification can leak in.
        """
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        # Resolve the scoped index first; no other certification enters the search.
        index, global_positions = self._scoped_index(certification_id)
        if not global_positions:
            return []
        vector = np.asarray(self.provider.embed_query(query), dtype="float32").reshape(1, -1)
        if vector.shape[1] != self.manifest["dimension"]:
            raise ValueError("Query embedding dimension does not match the stored index")
        vector = np.ascontiguousarray(vector)
        self._faiss.normalize_L2(vector)
        limit = min(top_k, len(global_positions))
        scores, local_indices = index.search(vector, limit)
        results: list[RetrievalResult] = []
        for score, local_index in zip(scores[0], local_indices[0], strict=True):
            if local_index < 0:
                continue
            chunk = self._chunks[global_positions[int(local_index)]]
            if chunk.certification_id != certification_id:
                raise RuntimeError("Certification index integrity check failed")
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    certification_id=chunk.certification_id,
                    certification=chunk.certification,
                    text=chunk.text,
                    topic=chunk.topic,
                    domain=chunk.domain,
                    subtopics=chunk.subtopics,
                    source=SourceReference(
                        source_url=chunk.source_url,
                        source_title=chunk.source_title,
                        source_type=chunk.source_type,
                    ),
                    similarity_score=round(float(score), 6),
                    content_detail_level=chunk.content_detail_level,
                )
            )
        return results


def retrieve_context(
    query: str,
    certification_id: str,
    top_k: int = 5,
    *,
    vector_store_dir: str | Path = "vector_store",
    embedding_provider: EmbeddingProvider | None = None,
) -> list[RetrievalResult]:
    """Convenience function with the requested retrieval signature."""

    return FaissRetriever(
        vector_store_dir, embedding_provider=embedding_provider
    ).retrieve_context(query, certification_id, top_k)


def corpus_quality_report(chunks: Sequence[RAGChunk]) -> dict[str, Any]:
    """Per-certification readiness report over the RAG corpus.

    A certification is 'question_generation_ready' only when every one of its chunks
    is detailed content (not a bare topic label), since thin context can't ground
    reliable questions. This is what marks an exam ready/not-ready in the catalog.
    """
    by_certification: dict[str, dict[str, Any]] = {}
    for certification_id in sorted({chunk.certification_id for chunk in chunks}):
        scoped = [chunk for chunk in chunks if chunk.certification_id == certification_id]
        detailed = sum(chunk.content_detail_level == "detailed" for chunk in scoped)
        generation_ready = bool(scoped) and detailed == len(scoped)
        by_certification[certification_id] = {
            "chunks": len(scoped),
            "detailed_chunks": detailed,
            "topic_label_chunks": len(scoped) - detailed,
            "question_generation_ready": generation_ready,
            "warning": ""
            if generation_ready
            else "The certification corpus is empty, incomplete, or includes scope-label chunks; detailed official content is required for reliable MCQs.",
        }
    return by_certification


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--certifications-dir", default="certifications")
    prepare_parser.add_argument("--chunks-output", default="data/rag_chunks.json")
    embed_parser = subparsers.add_parser("embed")
    embed_parser.add_argument("--chunks", default="data/rag_chunks.json")
    embed_parser.add_argument("--output-dir", default="vector_store")
    args = parser.parse_args()

    if args.command == "prepare":
        _, stats = prepare_rag_data(args.certifications_dir, args.chunks_output)
        print(json.dumps(stats, indent=2))
    else:
        raw_chunks = json.loads(Path(args.chunks).read_text(encoding="utf-8"))
        manifest = build_vector_store(raw_chunks, args.output_dir)
        print(
            json.dumps(
                {
                    key: value
                    for key, value in manifest.items()
                    if key not in {"chunks", "certifications"}
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    _main()
