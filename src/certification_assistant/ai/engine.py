"""Grounded OpenAI features plus deterministic assessment intelligence."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any, NamedTuple

from dotenv import load_dotenv
from pydantic import ValidationError as PydanticValidationError

from ..assessment.evaluation import calculate_readiness, detect_weak_areas, evaluate_assessment
from ..schemas import (
    AnswerEvaluation,
    AssessmentResult,
    ExplanationOutput,
    PracticeQuestion,
    PracticeQuestionBatch,
    QuestionGroundingVerdict,
    ReadinessAssessment,
    Recommendation,
    RecommendationBatch,
    RetrievalResult,
    RAGChunk,
    SemanticGroundingBatch,
    StudyPlan,
    StudyPlanDay,
    WeakArea,
    WeakAreaAnalysis,
)


DEFAULT_GENERATION_MODEL = "gpt-5-mini"
REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")
VALID_REASONING_EFFORTS = {"minimal", "low", "medium", "high"}
# Measured on gpt-5-mini writing one question per shard: "low" halves the time per call
# against "medium" and fails grounding more often, but the spare shards absorb those
# failures, so the batch lands in ~13s instead of ~35s. "minimal" is faster still and
# fails too often for the spares to cover on small batches. Raise this to "medium" via
# OPENAI_GENERATION_EFFORT to buy the model more deliberation per question.
DEFAULT_GENERATION_EFFORT = "low"
# The auditor is a verification pass, not a writing pass, and it reads one shard's
# question against one shard's context. Checked against a known-unsupported question:
# "minimal" rejected it 5/5 and approved the supported control 5/5, matching "low" and
# "medium" exactly, so the extra reasoning budget was buying nothing but latency.
DEFAULT_JUDGE_EFFORT = "minimal"
# Questions per concurrent generation call. A call's latency is dominated by how much
# it has to write, so small shards are what turns a one-call batch into a fan-out that
# finishes in the time of its slowest shard.
DEFAULT_GENERATION_SHARD_SIZE = 1
# Spare shards launched with the required ones, as a fraction of the required count.
# Measured on a 10-question batch: one shard failing grounding and retrying serially
# cost more wall clock than every other shard put together, so buying width up front
# is cheaper in time than paying for that tail. At the default "low" writing effort a
# shard fails often enough that one spare per required shard is what keeps the batch in
# a single wave; that roughly doubles generation calls, which is the price of the
# latency. Lower it if the account's rate limit, not latency, is the binding constraint.
DEFAULT_GENERATION_OVERPROVISION = 1.0
# Small batches need the floor, not the ratio: needing 3 successes out of 6 shards is a
# far less certain bet than needing 10 out of 20, and losing it costs a whole extra wave.
MIN_SPARE_SHARDS = 4
WORD_PATTERN = re.compile(r"[a-z0-9][a-z0-9+.#/-]*", re.IGNORECASE)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


class AIEngineConfigurationError(RuntimeError):
    """Raised when a requested live OpenAI operation lacks configuration."""


class GroundingValidationError(ValueError):
    """Raised when model output references evidence outside retrieved context."""


@lru_cache(maxsize=1)
def _load_env() -> None:
    """Read .env once per process.

    ``load_dotenv`` re-parses the file from disk on every call and never overrides
    variables already in the environment, so repeating it on every helper call only
    added file I/O to the request path.
    """
    load_dotenv()


def _resolve_model(model: str | None = None) -> str:
    """Pick which OpenAI model to use: the explicit argument, else the
    OPENAI_GENERATION_MODEL env var, else the built-in default."""
    _load_env()
    return model or os.getenv("OPENAI_GENERATION_MODEL", DEFAULT_GENERATION_MODEL)


@lru_cache(maxsize=2)
def _shared_client(key: str) -> Any:
    """One process-wide client per API key.

    The client owns an HTTP connection pool, so reusing it lets the shards that run
    concurrently share warm TLS connections instead of each paying for a handshake.

    The retry and timeout are explicit rather than left to the SDK defaults: a batch
    fans out to dozens of concurrent requests, so a transient DNS or connection drop
    is likely enough to be worth absorbing, and an unbounded request would otherwise
    let one hung shard hold a worker thread for the life of the process.
    """
    from openai import OpenAI

    return OpenAI(api_key=key, max_retries=3, timeout=90.0)


def _resolve_client(client: Any | None = None) -> Any:
    """Return an OpenAI client to call the API with.

    If a client is injected (e.g. in tests) it is reused as-is. Otherwise the
    OPENAI_API_KEY is read from the environment and a real ``OpenAI`` client is
    created (and cached for reuse). Raises ``AIEngineConfigurationError`` when no key
    is configured, so live features fail with a clear message instead of a vague crash.
    """
    if client is not None:
        return client
    _load_env()
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise AIEngineConfigurationError(
            "OPENAI_API_KEY is required for this live OpenAI operation"
        )
    return _shared_client(key)


def _supports_reasoning(model: str) -> bool:
    """True for reasoning-capable models (gpt-5 / o-series) that accept a
    ``reasoning.effort`` parameter; other models must not receive it."""
    return model.startswith(REASONING_MODEL_PREFIXES)


def _validated_effort(value: str, variable: str) -> str:
    """Normalize and validate a reasoning-effort value (minimal/low/medium/high),
    raising a clear error naming the offending env var if it is invalid."""
    effort = value.strip().lower()
    if effort not in VALID_REASONING_EFFORTS:
        raise ValueError(
            f"{variable} must be one of {', '.join(sorted(VALID_REASONING_EFFORTS))}; got {value!r}"
        )
    return effort


def _generation_efforts(candidate_count: int) -> tuple[str, ...]:
    """Reasoning effort for each raced candidate.

    By default every candidate uses the same effort and differs only by sampling, which
    is what shortens the tail: two independent tries share one wall-clock window instead
    of costing two sequential rounds. OPENAI_GENERATION_EFFORTS can supply an explicit
    per-candidate ladder for experimentation.
    """

    _load_env()
    single = _validated_effort(
        os.getenv("OPENAI_GENERATION_EFFORT", DEFAULT_GENERATION_EFFORT),
        "OPENAI_GENERATION_EFFORT",
    )
    raw = os.getenv("OPENAI_GENERATION_EFFORTS", "").strip()
    if not raw:
        return (single,) * candidate_count
    ladder = tuple(
        _validated_effort(part, "OPENAI_GENERATION_EFFORTS")
        for part in raw.split(",")
        if part.strip()
    )
    if not ladder:
        return (single,) * candidate_count
    return tuple(ladder[index % len(ladder)] for index in range(candidate_count))


def _judge_effort() -> str:
    """Reasoning effort for the semantic grounding auditor (a verification pass,
    so it defaults to 'low' — configurable via OPENAI_JUDGE_EFFORT)."""
    _load_env()
    return _validated_effort(
        os.getenv("OPENAI_JUDGE_EFFORT", DEFAULT_JUDGE_EFFORT), "OPENAI_JUDGE_EFFORT"
    )


def _resolve_shard_size(shard_size: int | None = None) -> int:
    """Questions per concurrent generation call: the explicit argument, else the
    OPENAI_GENERATION_SHARD_SIZE env var, else the built-in default."""
    if shard_size is not None:
        if shard_size < 1:
            raise ValueError("shard_size must be at least 1")
        return shard_size
    _load_env()
    raw = os.getenv("OPENAI_GENERATION_SHARD_SIZE", "").strip()
    if not raw:
        return DEFAULT_GENERATION_SHARD_SIZE
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"OPENAI_GENERATION_SHARD_SIZE must be a positive integer; got {raw!r}"
        ) from exc
    if value < 1:
        raise ValueError(
            f"OPENAI_GENERATION_SHARD_SIZE must be a positive integer; got {raw!r}"
        )
    return value


def _parse_structured(
    client: Any,
    *,
    model: str,
    input: list[dict[str, str]],
    text_format: Any,
    effort: str | None,
) -> Any:
    """Call Responses.parse, adding reasoning effort only when the model accepts it."""

    kwargs: dict[str, Any] = {
        "model": model,
        "input": input,
        "text_format": text_format,
        "store": False,
    }
    if effort and _supports_reasoning(model):
        kwargs["reasoning"] = {"effort": effort}
    try:
        return client.responses.parse(**kwargs)
    except TypeError as exc:
        if "reasoning" not in str(exc) or "reasoning" not in kwargs:
            raise
    except Exception as exc:  # pragma: no cover - provider-specific rejection
        if "reasoning" not in str(exc).lower() or "reasoning" not in kwargs:
            raise
    kwargs.pop("reasoning", None)
    return client.responses.parse(**kwargs)


def _context_record(item: RetrievalResult | RAGChunk | Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one retrieved knowledge chunk (from any accepted input shape)
    into a uniform dict with the fields generation/grounding rely on: chunk_id,
    certification, domain, topic, text, and source URL/title/type."""
    if isinstance(item, RetrievalResult):
        return {
            "chunk_id": item.chunk_id,
            "certification_id": item.certification_id,
            "certification": item.certification,
            "domain": item.domain,
            "topic": item.topic,
            "subtopics": item.subtopics,
            "text": item.text,
            "source_url": item.source.source_url,
            "source_title": item.source.source_title,
            "source_type": item.source.source_type,
            "source_urls": [item.source.source_url] if item.source.source_url else [],
            "content_detail_level": item.content_detail_level,
        }
    if isinstance(item, RAGChunk):
        return item.model_dump(mode="json")
    raw = dict(item)
    source = raw.get("source", {})
    if isinstance(source, Mapping):
        raw.setdefault("source_url", source.get("source_url", source.get("url", "")))
        raw.setdefault("source_title", source.get("source_title", source.get("title", "")))
        raw.setdefault("source_type", source.get("source_type", source.get("type", "official")))
    raw.setdefault("source_urls", [raw["source_url"]] if raw.get("source_url") else [])
    raw.setdefault("subtopics", [])
    raw.setdefault("content_detail_level", "detailed")
    required = ("chunk_id", "certification", "domain", "topic", "text")
    missing = [field for field in required if not raw.get(field)]
    if missing:
        raise ValueError(f"Retrieved context is missing required fields: {', '.join(missing)}")
    return raw


def _context_records(
    items: Sequence[RetrievalResult | RAGChunk | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    # Normalize every item, then drop duplicates by chunk_id so the same source
    # chunk is never fed to the model (or counted) twice.
    records = [_context_record(item) for item in items]
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for record in records:
        if record["chunk_id"] in seen:
            continue
        seen.add(record["chunk_id"])
        unique.append(record)
    return unique


def context_is_sufficient(
    retrieved_knowledge_chunks: Sequence[RetrievalResult | RAGChunk | Mapping[str, Any]],
) -> bool:
    """Require actual explanatory content and an official URL, not scope labels alone."""

    for record in _context_records(retrieved_knowledge_chunks):
        if (
            record.get("content_detail_level") == "detailed"
            and len(str(record.get("text", ""))) >= 120
            and bool(record.get("source_url"))
        ):
            return True
    return False


def _question_prompt_context(records: Sequence[Mapping[str, Any]]) -> str:
    """Serialize up to 15 retrieved chunks into the compact JSON block embedded
    in the prompt as the model's ONLY allowed source of truth (text is capped at
    6000 chars per chunk to control token cost)."""
    safe_records = [
        {
            "chunk_id": record["chunk_id"],
            "certification": record["certification"],
            "domain": record["domain"],
            "topic": record["topic"],
            "subtopics": record.get("subtopics", []),
            "text": str(record["text"])[:6000],
            "source_url": record.get("source_url", ""),
        }
        for record in records[:15]
    ]
    return json.dumps(safe_records, ensure_ascii=False, indent=2)


def validate_question_grounding(
    question: PracticeQuestion | Mapping[str, Any],
    retrieved_knowledge_chunks: Sequence[RetrievalResult | RAGChunk | Mapping[str, Any]],
) -> QuestionGroundingVerdict:
    """Apply deterministic provenance and conservative lexical-support checks."""

    parsed = (
        question
        if isinstance(question, PracticeQuestion)
        else PracticeQuestion.model_validate(question)
    )
    records = _context_records(retrieved_knowledge_chunks)
    by_id = {record["chunk_id"]: record for record in records}
    unknown_ids = [identifier for identifier in parsed.source_chunk_ids if identifier not in by_id]
    if unknown_ids:
        return QuestionGroundingVerdict(
            supported=False,
            all_options_grounded=False,
            correct_answer_supported=False,
            explanation_supported=False,
            unsupported_options=["A", "B", "C", "D"],
            reason=f"Unknown source chunk ids: {', '.join(unknown_ids)}",
        )
    cited_records = [by_id[identifier] for identifier in parsed.source_chunk_ids]
    allowed_urls = {
        url
        for record in cited_records
        for url in [record.get("source_url", ""), *record.get("source_urls", [])]
        if url
    }
    if not set(parsed.source_urls).issubset(allowed_urls):
        return QuestionGroundingVerdict(
            supported=False,
            all_options_grounded=False,
            correct_answer_supported=False,
            explanation_supported=False,
            unsupported_options=["A", "B", "C", "D"],
            reason="One or more source URLs were not present in the cited chunks.",
        )
    allowed_domains = {record["domain"] for record in cited_records}
    allowed_topics = {record["topic"] for record in cited_records}
    if parsed.domain not in allowed_domains or parsed.topic not in allowed_topics:
        return QuestionGroundingVerdict(
            supported=False,
            all_options_grounded=False,
            correct_answer_supported=False,
            explanation_supported=False,
            unsupported_options=["A", "B", "C", "D"],
            reason="Question domain or topic does not match its cited chunks.",
        )

    context_text = " ".join(str(record["text"]).casefold() for record in cited_records)
    context_tokens = {token.casefold() for token in WORD_PATTERN.findall(context_text)}

    def option_is_grounded(option_text: str) -> bool:
        normalized = " ".join(option_text.casefold().split())
        option_tokens = {
            token.casefold()
            for token in WORD_PATTERN.findall(normalized)
            if token.casefold() not in STOPWORDS and len(token) > 2
        }
        if not option_tokens:
            return False
        # Exact context phrases are strongest. The high token-overlap fallback allows
        # harmless grammatical variation while rejecting outside-knowledge distractors.
        return normalized in context_text or (
            len(option_tokens & context_tokens) / len(option_tokens) >= 0.80
        )

    option_support = {
        label: option_is_grounded(option)
        for label, option in parsed.options.items()
    }
    unsupported_options = [
        label for label, grounded in option_support.items() if not grounded
    ]
    all_options_grounded = not unsupported_options
    answer_supported = option_support[parsed.correct_answer]
    explanation_tokens = {
        token.casefold()
        for token in WORD_PATTERN.findall(parsed.explanation)
        if token.casefold() not in STOPWORDS and len(token) > 3
    }
    explanation_supported = bool(explanation_tokens) and (
        len(explanation_tokens & context_tokens) / len(explanation_tokens) >= 0.30
    )
    supported = all_options_grounded and answer_supported and explanation_supported
    if unsupported_options:
        reason = (
            "Unsupported answer option(s): "
            + ", ".join(unsupported_options)
            + ". Every option must use concepts present in the cited context."
        )
    elif not answer_supported:
        reason = "The correct answer is not supported by the cited context."
    elif not explanation_supported:
        reason = "The explanation is not supported strongly enough by the cited context."
    else:
        reason = "Deterministic citation and lexical checks passed for all options, the answer, and the explanation."
    return QuestionGroundingVerdict(
        supported=supported,
        all_options_grounded=all_options_grounded,
        correct_answer_supported=answer_supported,
        explanation_supported=explanation_supported,
        unsupported_options=unsupported_options,
        reason=reason,
    )


def review_question_grounding_semantically(
    questions: Sequence[PracticeQuestion],
    retrieved_knowledge_chunks: Sequence[
        RetrievalResult | RAGChunk | Mapping[str, Any]
    ],
    *,
    client: Any | None = None,
    model: str | None = None,
    effort: str | None = None,
) -> SemanticGroundingBatch:
    """Use a closed-book entailment review after deterministic provenance checks."""

    records = _context_records(retrieved_knowledge_chunks)
    allowed_ids = {record["chunk_id"] for record in records}
    response = _parse_structured(
        _resolve_client(client),
        model=_resolve_model(model),
        effort=effort or _judge_effort(),
        input=[
            {
                "role": "system",
                "content": (
                    "You are a strict closed-book provenance auditor. Treat RETRIEVED CONTEXT "
                    "as the entire universe of allowed knowledge, even when you personally know "
                    "more. A question is supported only when its tested relationship and every "
                    "claim in its explanation are explicitly stated or directly entailed by the "
                    "cited context. Reject unstated definitions, properties, cause/effect claims, "
                    "comparisons, service capabilities, best-choice mappings, or term-to-behavior "
                    "relationships. Merely seeing both words in context is not support. Verify all "
                    "four options are context-grounded concepts and the answer is uniquely correct. "
                    "Evidence chunk IDs must come from RETRIEVED CONTEXT."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "retrieved_context": json.loads(_question_prompt_context(records)),
                        "questions": [
                            question.model_dump(mode="json") for question in questions
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ],
        text_format=SemanticGroundingBatch,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise GroundingValidationError("OpenAI returned no semantic grounding review")
    review = SemanticGroundingBatch.model_validate(parsed)
    expected_numbers = list(range(1, len(questions) + 1))
    actual_numbers = [item.question_number for item in review.reviews]
    if actual_numbers != expected_numbers:
        raise GroundingValidationError(
            "Semantic grounding review did not return one ordered result per question"
        )
    for item in review.reviews:
        if not set(item.evidence_chunk_ids).issubset(allowed_ids):
            raise GroundingValidationError(
                "Semantic grounding review cited an unknown chunk ID"
            )
        if item.supported and not item.evidence_chunk_ids:
            raise GroundingValidationError(
                "Semantic grounding review approved a question without evidence"
            )
    computed_all_supported = all(item.supported for item in review.reviews)
    if review.all_supported != computed_all_supported:
        raise GroundingValidationError(
            "Semantic grounding review returned inconsistent aggregate status"
        )
    return review


def _normalized_batch(result: PracticeQuestionBatch) -> PracticeQuestionBatch:
    """De-duplicate each question's cited chunk IDs and source URLs (order-preserving)
    so provenance lists are clean before validation."""
    return result.model_copy(
        update={
            "questions": [
                question.model_copy(
                    update={
                        "source_chunk_ids": list(dict.fromkeys(question.source_chunk_ids)),
                        "source_urls": list(dict.fromkeys(question.source_urls)),
                    }
                )
                for question in result.questions
            ]
        }
    )


def _generate_candidate(
    client: Any,
    model: str,
    effort: str,
    system_prompt: str,
    user_prompt: str,
) -> tuple[PracticeQuestionBatch | None, PydanticValidationError | None]:
    """One generation call. Returns (batch, schema_error); batch is None on schema failure."""

    try:
        response = _parse_structured(
            client,
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text_format=PracticeQuestionBatch,
            effort=effort,
        )
    except PydanticValidationError as exc:
        return None, exc
    parsed_output = response.output_parsed
    if parsed_output is None:
        return None, None
    return _normalized_batch(PracticeQuestionBatch.model_validate(parsed_output)), None


def _deterministic_failures(
    result: PracticeQuestionBatch,
    certification: str,
    difficulty: str,
    number_of_questions: int,
    records: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Local, free provenance checks. Identical rules to the previous serial path."""

    failures: list[str] = []
    if len(result.questions) != number_of_questions:
        failures.append("Return exactly the requested question count or insufficient_context.")
    seen_questions: set[str] = set()
    for index, question in enumerate(result.questions, start=1):
        if question.certification != certification or question.difficulty != difficulty:
            failures.append(
                f"Question {index}: certification or difficulty differs from the request."
            )
        normalized_question = " ".join(question.question.casefold().split())
        if normalized_question in seen_questions:
            failures.append(f"Question {index}: duplicate question text.")
        seen_questions.add(normalized_question)
        verdict = validate_question_grounding(question, records)
        if not verdict.supported:
            failures.append(f"Question {index}: {verdict.reason}")
    return failures


class _CandidateOutcome(NamedTuple):
    """Result of one full generate -> validate -> audit pass."""

    batch: PracticeQuestionBatch | None
    failures: list[str]
    schema_error: PydanticValidationError | None
    insufficient: PracticeQuestionBatch | None
    empty_output: bool

    @property
    def accepted(self) -> bool:
        return self.batch is not None and not self.failures


def _evaluate_candidate(
    client: Any,
    model: str | None,
    resolved_model: str,
    effort: str,
    system_prompt: str,
    user_prompt: str,
    certification: str,
    difficulty: str,
    number_of_questions: int,
    records: Sequence[Mapping[str, Any]],
    judge_effort: str,
    use_semantic_grounding_judge: bool,
) -> _CandidateOutcome:
    """Run every grounding check for one candidate so racing them is meaningful."""

    candidate, schema_error = _generate_candidate(
        client, resolved_model, effort, system_prompt, user_prompt
    )
    if schema_error is not None:
        return _CandidateOutcome(None, [], schema_error, None, False)
    if candidate is None:
        return _CandidateOutcome(None, [], None, None, True)
    if candidate.status == "insufficient_context":
        return _CandidateOutcome(None, [], None, candidate, False)
    failures = _deterministic_failures(
        candidate, certification, difficulty, number_of_questions, records
    )
    if failures:
        return _CandidateOutcome(candidate, failures, None, None, False)
    if not use_semantic_grounding_judge:
        return _CandidateOutcome(candidate, [], None, None, False)
    semantic = _semantic_failures(candidate, records, client, model, judge_effort)
    return _CandidateOutcome(candidate, semantic, None, None, False)


def _semantic_failures(
    candidate: PracticeQuestionBatch,
    records: Sequence[Mapping[str, Any]],
    client: Any,
    model: str | None,
    effort: str,
) -> list[str]:
    """Run the LLM 'closed-book auditor' over a candidate batch and return a list
    of human-readable rejection reasons (empty list = every question passed)."""
    try:
        semantic_review = review_question_grounding_semantically(
            candidate.questions, records, client=client, model=model, effort=effort
        )
    except (GroundingValidationError, PydanticValidationError) as exc:
        return [f"Semantic grounding review failed: {exc}"]
    failures: list[str] = []
    for item in semantic_review.reviews:
        if not item.supported:
            claims = "; ".join(item.unsupported_claims) or item.reason
            failures.append(
                f"Question {item.question_number}: semantic grounding review "
                f"rejected unsupported claim(s): {claims}"
            )
    return failures


class _ShardOutcome(NamedTuple):
    """Result of one shard's full generate -> validate -> audit -> retry cycle."""

    questions: list[PracticeQuestion]
    insufficient: PracticeQuestionBatch | None
    error: str | None
    schema_failure: bool
    empty_output: bool


def _shard_sizes(total: int, shard_size: int) -> list[int]:
    """Split a requested question count into as-even-as-possible parallel shards.

    A generation call's latency scales with how many questions the model has to write,
    so N questions spread over ceil(N/shard_size) concurrent calls costs roughly the
    time of the slowest shard instead of the whole batch written serially.
    """

    shard_size = max(1, shard_size)
    if total <= shard_size:
        return [total]
    count = -(-total // shard_size)
    base, extra = divmod(total, count)
    return [base + 1] * extra + [base] * (count - extra)


def _shard_topics(records: Sequence[Mapping[str, Any]], shard_count: int) -> list[list[str]]:
    """Deal the retrieved topics round-robin across shards.

    Shards run blind to each other, so without a disjoint topic assignment two of them
    routinely write the same question. When shards outnumber topics the extra shards
    double up on a topic rather than falling back to the whole context: that only costs
    a duplicate that merge-time de-duplication drops, whereas widening the context back
    out would cost every shard the larger prompt.
    """

    topics = list(dict.fromkeys(str(record["topic"]) for record in records))
    if shard_count < 2 or not topics:
        return [[] for _ in range(shard_count)]
    groups: list[list[str]] = [[] for _ in range(shard_count)]
    if len(topics) >= shard_count:
        for index, topic in enumerate(topics):
            groups[index % shard_count].append(topic)
    else:
        for index in range(shard_count):
            groups[index].append(topics[index % len(topics)])
    return groups


def _shard_records(
    records: Sequence[Mapping[str, Any]], topics: Sequence[str]
) -> list[dict[str, Any]]:
    """Narrow the context to a shard's assigned topics.

    A shard may only write about the topics it was dealt, so shipping it the other
    topics' chunks only inflates the prompt it has to read. Falls back to the full
    context whenever the narrowed slice would not clear the sufficiency bar.
    """

    if not topics:
        return list(records)
    wanted = set(topics)
    slice_ = [record for record in records if str(record["topic"]) in wanted]
    return slice_ if context_is_sufficient(slice_) else list(records)


def _question_key(question: PracticeQuestion) -> str:
    """Whitespace/case-insensitive question text, used to drop cross-shard duplicates."""

    return " ".join(question.question.casefold().split())


def _overprovision(shard_count: int) -> int:
    """How many spare shards to launch alongside the required ones.

    Shards fail grounding independently, and the profile of a full batch showed a
    single straggler retrying serially for longer than every other shard combined.
    Spare shards convert that retry tail into extra width: the first shards to clear
    validation supply the batch and the rest are abandoned.
    """

    if shard_count < 2:
        return 0
    _load_env()
    raw = os.getenv("OPENAI_GENERATION_OVERPROVISION", "").strip()
    try:
        ratio = float(raw) if raw else DEFAULT_GENERATION_OVERPROVISION
    except ValueError as exc:
        raise ValueError(
            f"OPENAI_GENERATION_OVERPROVISION must be a number; got {raw!r}"
        ) from exc
    if ratio <= 0:
        return 0
    # The floor matters more than the ratio on small batches: a 3-question exam that
    # loses one shard has no slack at all, while a 10-question one can absorb it.
    return max(MIN_SPARE_SHARDS, round(shard_count * ratio))


def _shard_user_prompt(
    certification: str,
    difficulty: str,
    number_of_questions: int,
    context_block: str,
    topics: Sequence[str],
    avoid_questions: Sequence[str],
) -> str:
    """Build one shard's user prompt: the shared instructions plus its own slice."""

    prompt = (
        f"Certification: {certification}\n"
        f"Difficulty: {difficulty}\n"
        f"Requested question count: {number_of_questions}\n"
        "Create exactly the requested number if the context supports them; otherwise return "
        "insufficient_context with no questions.\n"
    )
    if topics:
        prompt += (
            "Restrict this batch to these assigned topics only, so it stays distinct from the "
            "batches being written in parallel: " + "; ".join(topics) + "\n"
        )
    if avoid_questions:
        prompt += (
            "These questions already exist; do not repeat or paraphrase them: "
            + " | ".join(avoid_questions)
            + "\n"
        )
    return prompt + f"\nRETRIEVED CONTEXT\n{context_block}"


def _run_shard(
    client: Any,
    model: str | None,
    resolved_model: str,
    efforts: tuple[str, ...],
    judge_effort: str,
    system_prompt: str,
    user_prompt: str,
    certification: str,
    difficulty: str,
    number_of_questions: int,
    records: Sequence[Mapping[str, Any]],
    use_semantic_grounding_judge: bool,
    max_grounding_attempts: int,
    feedback: str = "",
) -> _ShardOutcome:
    """Generate and fully validate one shard, retrying only this shard on failure.

    Each attempt races ``efforts`` independent candidates and keeps the first that clears
    every grounding check, so a rejected candidate costs no extra wall-clock time.
    ``feedback`` carries a previous wave's rejection reason into the first attempt.
    """

    last_failure = feedback or "Grounding validation failed."
    schema_failure_seen = False

    for _attempt in range(1, max_grounding_attempts + 1):
        attempt_prompt = user_prompt
        if feedback:
            attempt_prompt += (
                "\n\nA previous attempt failed deterministic grounding validation. "
                f"Fix this issue and regenerate the complete batch: {feedback}"
            )

        round_failures: list[str] = []
        insufficient: PracticeQuestionBatch | None = None
        empty_output = False
        # Not a `with` block: returning the winner must not block on the losing candidates.
        pool = ThreadPoolExecutor(max_workers=len(efforts))
        try:
            futures = [
                pool.submit(
                    _evaluate_candidate,
                    client,
                    model,
                    resolved_model,
                    effort,
                    system_prompt,
                    attempt_prompt,
                    certification,
                    difficulty,
                    number_of_questions,
                    records,
                    judge_effort,
                    use_semantic_grounding_judge,
                )
                for effort in efforts
            ]
            for future in as_completed(futures):
                outcome = future.result()
                if outcome.accepted:
                    return _ShardOutcome(list(outcome.batch.questions), None, None, False, False)
                if outcome.schema_error is not None:
                    schema_failure_seen = True
                    round_failures.append(
                        "The structured output did not satisfy the Pydantic schema: "
                        + str(outcome.schema_error).replace("\n", " ")[:1200]
                    )
                if outcome.insufficient is not None:
                    insufficient = outcome.insufficient
                if outcome.empty_output:
                    empty_output = True
                round_failures.extend(outcome.failures)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        # A candidate reporting insufficient context is a terminal answer, not something
        # a retry can fix, so it wins only when nothing else produced usable questions.
        if insufficient is not None:
            return _ShardOutcome([], insufficient, None, False, False)
        if empty_output and not round_failures:
            return _ShardOutcome([], None, None, False, True)
        if round_failures:
            last_failure = " ".join(round_failures)
            feedback = last_failure[:4000]

    return _ShardOutcome([], None, last_failure, schema_failure_seen, False)


def generate_practice_questions(
    certification: str,
    difficulty: str,
    number_of_questions: int,
    retrieved_knowledge_chunks: Sequence[RetrievalResult | RAGChunk | Mapping[str, Any]],
    *,
    client: Any | None = None,
    model: str | None = None,
    max_grounding_attempts: int = 3,
    use_semantic_grounding_judge: bool = True,
    parallel_candidates: int = 1,
    shard_size: int | None = None,
) -> PracticeQuestionBatch:
    """Generate new, cited practice MCQs with OpenAI Structured Outputs.

    The batch is written as concurrent shards of at most ``shard_size`` questions, each
    reading only its own assigned topics, so wall-clock time tracks the slowest shard
    rather than the question count. Spare shards are launched alongside the required
    ones and the first questions to clear every grounding check win, which keeps one
    unlucky shard from serialising retries onto the end of the request; anything still
    missing is topped up in a second, feedback-carrying wave. ``parallel_candidates``
    additionally races that many independent generations inside each shard attempt.
    Validation rules are unchanged; only the scheduling of the work differs.
    """

    if difficulty not in {"easy", "medium", "hard"}:
        raise ValueError("difficulty must be easy, medium, or hard")
    if not 1 <= number_of_questions <= 20:
        raise ValueError("number_of_questions must be between 1 and 20")
    if not 1 <= max_grounding_attempts <= 3:
        raise ValueError("max_grounding_attempts must be between 1 and 3")
    if not 1 <= parallel_candidates <= 5:
        raise ValueError("parallel_candidates must be between 1 and 5")
    records = _context_records(retrieved_knowledge_chunks)
    if not records:
        return PracticeQuestionBatch(
            status="insufficient_context",
            reason="No retrieved context was supplied.",
            questions=[],
        )
    if any(record["certification"] != certification for record in records):
        raise GroundingValidationError(
            "All retrieved chunks must match the requested certification"
        )
    if not context_is_sufficient(records):
        return PracticeQuestionBatch(
            status="insufficient_context",
            reason=(
                "Retrieved records contain only scope/topic labels or lack an official source URL. "
                "Detailed official content is required for reliable questions."
            ),
            questions=[],
        )

    system_prompt = (
        "You create original certification practice questions, never leaked or real exam items. "
        "Use only the supplied retrieved context. If the context is insufficient to create a "
        "reliable question, return insufficient_context instead of inventing information. "
        "Every factual claim, correct answer, and explanation must be directly supported by the "
        "cited source_chunk_ids. Cite only chunk IDs and source URLs supplied in context. Set "
        "practice_only to true. All four answer option texts must use only concepts, terms, "
        "services, or factual statements that appear in the cited context. Distractors must be "
        "plausible but cannot introduce a concept or claim from outside the cited context. Do not "
        "use outside knowledge. The relationship tested by the question stem must itself be "
        "explicitly stated in context. Do not infer an unstated definition, behavior, benefit, "
        "cause, comparison, or best-use mapping merely because the relevant terms are listed. "
        "Use short answer options copied verbatim from the cited context; before returning, verify "
        "that every complete option phrase occurs in a cited chunk."
    )
    openai_client = _resolve_client(client)
    resolved_model = _resolve_model(model)
    efforts = _generation_efforts(parallel_candidates)
    judge_effort = _judge_effort()
    resolved_shard_size = _resolve_shard_size(shard_size)

    accepted: list[PracticeQuestion] = []
    seen: set[str] = set()
    insufficient_batch: PracticeQuestionBatch | None = None
    last_failure = "Grounding validation failed."
    schema_failure_seen = False
    empty_output_seen = False

    def launch(
        sizes: Sequence[int],
        topic_groups: Sequence[Sequence[str]],
        attempts: int,
        feedback: str,
    ) -> None:
        """Run one wave of shards concurrently, folding results in as they land."""

        nonlocal insufficient_batch, last_failure, schema_failure_seen, empty_output_seen
        avoid = [question.question for question in accepted]
        assignments = [
            (count, topics, _shard_records(records, topics))
            for count, topics in zip(sizes, topic_groups, strict=True)
        ]
        # Not a `with` block: once the batch is complete the remaining shards are
        # abandoned rather than waited on.
        pool = ThreadPoolExecutor(max_workers=len(sizes))
        try:
            futures = [
                pool.submit(
                    _run_shard,
                    openai_client,
                    model,
                    resolved_model,
                    efforts,
                    judge_effort,
                    system_prompt,
                    _shard_user_prompt(
                        certification,
                        difficulty,
                        count,
                        _question_prompt_context(shard_records),
                        topics,
                        avoid,
                    ),
                    certification,
                    difficulty,
                    count,
                    shard_records,
                    use_semantic_grounding_judge,
                    attempts,
                    feedback,
                )
                for count, topics, shard_records in assignments
            ]
            for future in as_completed(futures):
                outcome = future.result()
                if outcome.insufficient is not None:
                    insufficient_batch = outcome.insufficient
                if outcome.error:
                    last_failure = outcome.error
                schema_failure_seen = schema_failure_seen or outcome.schema_failure
                empty_output_seen = empty_output_seen or outcome.empty_output
                for question in outcome.questions:
                    key = _question_key(question)
                    if key in seen:
                        continue
                    seen.add(key)
                    accepted.append(question)
                if len(accepted) >= number_of_questions:
                    return
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    required = _shard_sizes(number_of_questions, resolved_shard_size)
    spare = _overprovision(len(required))
    sizes = list(required) + [resolved_shard_size] * spare
    launch(sizes, _shard_topics(records, len(sizes)), 1, "")

    # A second wave only runs when the first came up short — every spare shard failed
    # grounding, or de-duplication collapsed two shards onto the same question. It
    # carries the rejection reason forward and drops the topic restriction so the
    # remaining questions can come from anywhere in the retrieved context.
    missing = number_of_questions - len(accepted)
    if missing > 0 and max_grounding_attempts > 1 and insufficient_batch is None:
        top_up = _shard_sizes(missing, resolved_shard_size)
        # Spares here too: this wave is the one that decides how bad a slow request gets,
        # so it buys width first and falls back on its serial retries only after that.
        top_up += [resolved_shard_size] * _overprovision(max(2, len(top_up)))
        launch(top_up, [[] for _ in top_up], max_grounding_attempts - 1, last_failure[:4000])

    if len(accepted) >= number_of_questions:
        return PracticeQuestionBatch(status="ok", questions=accepted[:number_of_questions])
    # Insufficient context is a terminal verdict, so it only loses to a batch that
    # actually produced every requested question.
    if insufficient_batch is not None:
        return insufficient_batch
    if empty_output_seen:
        raise RuntimeError(
            "OpenAI returned no parsed output (possibly a refusal or incomplete response)"
        )
    if schema_failure_seen:
        raise GroundingValidationError(
            f"OpenAI output failed schema validation after {max_grounding_attempts} attempt(s): "
            f"{last_failure}"
        )
    raise GroundingValidationError(
        f"OpenAI output failed grounding validation after {max_grounding_attempts} attempt(s): "
        f"{last_failure}"
    )


def evaluate_answer(
    question: PracticeQuestion | Mapping[str, Any], user_answer: str
) -> AnswerEvaluation:
    """Evaluate one MCQ deterministically using its known answer key."""

    parsed = (
        question
        if isinstance(question, PracticeQuestion)
        else PracticeQuestion.model_validate(question)
    )
    selected = str(user_answer).strip().upper()
    return AnswerEvaluation(
        is_correct=selected == parsed.correct_answer,
        selected_answer=selected,
        correct_answer=parsed.correct_answer,
        explanation=parsed.explanation,
    )


def analyze_weak_areas(
    assessment_results: AssessmentResult | Mapping[str, Any],
    *,
    topic_domains: Mapping[str, str] | None = None,
    topic_source_urls: Mapping[str, Sequence[str]] | None = None,
    review_threshold: float = 80.0,
    explain_with_openai: bool = False,
    client: Any | None = None,
    model: str | None = None,
) -> WeakAreaAnalysis:
    """Identify weak topics from a scored assessment.

    The weak-area detection itself is deterministic (``detect_weak_areas`` — topics
    below the review threshold, ranked). OpenAI is optional and only writes a short
    natural-language explanation of that fixed result; it never changes the scores,
    adds topics, or invents facts.
    """
    analysis = detect_weak_areas(
        assessment_results,
        topic_domains=topic_domains,
        topic_source_urls=topic_source_urls,
        review_threshold=review_threshold,
    )
    if not explain_with_openai or not analysis.weak_areas:
        return analysis
    response = _resolve_client(client).responses.parse(
        model=_resolve_model(model),
        input=[
            {
                "role": "system",
                "content": (
                    "Explain the supplied deterministic performance result concisely. Do not "
                    "change scores, add topics, or introduce certification facts."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(analysis.model_dump(mode="json"), ensure_ascii=False),
            },
        ],
        text_format=ExplanationOutput,
        store=False,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise RuntimeError("OpenAI returned no parsed weak-area explanation")
    return analysis.model_copy(update={"explanation": parsed.explanation})


def _official_resource_map(
    resources: Sequence[RetrievalResult | RAGChunk | Mapping[str, Any]],
) -> tuple[dict[str, list[str]], set[str]]:
    """Build (topic -> official source URLs) plus the flat set of all allowed URLs
    from official chunks only. Used so recommendations/study plans can cite only
    real official links and never invent a resource."""
    topic_urls: dict[str, list[str]] = {}
    all_urls: set[str] = set()
    for record in _context_records(resources):
        if not str(record.get("source_type", "official")).casefold().startswith("official"):
            continue
        urls = [record.get("source_url", ""), *record.get("source_urls", [])]
        clean_urls = [str(url) for url in urls if url]
        topic_urls.setdefault(record["topic"], [])
        topic_urls[record["topic"]].extend(clean_urls)
        topic_urls[record["topic"]] = list(dict.fromkeys(topic_urls[record["topic"]]))
        all_urls.update(clean_urls)
    return topic_urls, all_urls


def generate_recommendations(
    weak_areas: WeakAreaAnalysis | Sequence[WeakArea | Mapping[str, Any]],
    assessment_score: float,
    certification: str,
    retrieved_official_resources: Sequence[
        RetrievalResult | RAGChunk | Mapping[str, Any]
    ],
    *,
    use_openai: bool = True,
    client: Any | None = None,
    model: str | None = None,
) -> RecommendationBatch:
    """Build targeted study recommendations from the learner's weak areas.

    A deterministic draft is created first: one recommendation per weak topic, with
    its priority, an official source URL, and a reason citing the measured accuracy.
    If ``use_openai`` is set, OpenAI only rewrites the wording — guarded afterwards so
    it cannot change the certification, add an unknown topic, or cite a URL that was
    not in the supplied official resources.
    """
    if not 0 <= assessment_score <= 100:
        raise ValueError("assessment_score must be between 0 and 100")
    raw_areas = weak_areas.weak_areas if isinstance(weak_areas, WeakAreaAnalysis) else weak_areas
    areas = [
        area if isinstance(area, WeakArea) else WeakArea.model_validate(area)
        for area in raw_areas
    ]
    topic_urls, allowed_urls = _official_resource_map(retrieved_official_resources)
    deterministic = RecommendationBatch(
        certification=certification,
        recommendations=[
            Recommendation(
                what_to_study=area.topic,
                priority=area.priority,
                recommended_source=(
                    topic_urls.get(area.topic, [])
                    or [url for url in area.source_urls if url in allowed_urls]
                    or [""]
                )[0],
                reason=(
                    f"Accuracy for {area.topic} was {area.accuracy:.2f}% on an overall "
                    f"assessment score of {assessment_score:.2f}%."
                ),
            )
            for area in areas
        ],
    )
    if not use_openai or not deterministic.recommendations:
        return deterministic

    response = _resolve_client(client).responses.parse(
        model=_resolve_model(model),
        input=[
            {
                "role": "system",
                "content": (
                    "Create concise study recommendations only from the supplied weak areas and "
                    "official URLs. Preserve the certification, priorities, scores, and topic "
                    "names. Never add a URL or resource not supplied."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "certification": certification,
                        "assessment_score": assessment_score,
                        "draft": deterministic.model_dump(mode="json"),
                        "allowed_official_urls": sorted(allowed_urls),
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        text_format=RecommendationBatch,
        store=False,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise RuntimeError("OpenAI returned no parsed recommendations")
    result = RecommendationBatch.model_validate(parsed)
    if result.certification != certification:
        raise GroundingValidationError("Recommendation certification changed")
    if any(
        recommendation.recommended_source
        and recommendation.recommended_source not in allowed_urls
        for recommendation in result.recommendations
    ):
        raise GroundingValidationError("Recommendation included an unsupported resource URL")
    if not {item.what_to_study for item in result.recommendations}.issubset(
        {area.topic for area in areas}
    ):
        raise GroundingValidationError("Recommendation included an unsupported topic")
    return result


def generate_study_plan(
    certification: str,
    weak_areas: WeakAreaAnalysis | Sequence[WeakArea | Mapping[str, Any]],
    available_days: int,
    *,
    retrieved_official_resources: Sequence[
        RetrievalResult | RAGChunk | Mapping[str, Any]
    ] = (),
    use_openai: bool = False,
    client: Any | None = None,
    model: str | None = None,
) -> StudyPlan:
    """Build a day-by-day study plan prioritizing the weakest, highest-priority topics.

    Deterministic: weak areas are sorted by priority then accuracy, then spread across
    the available days (cycling through topics), each day carrying its topic, priority,
    fixed activities, and official resource links. OpenAI is optional and may only
    polish wording — it is rejected if it alters day numbers, topics, priorities, or URLs.
    """
    if not 1 <= available_days <= 365:
        raise ValueError("available_days must be between 1 and 365")
    raw_areas = weak_areas.weak_areas if isinstance(weak_areas, WeakAreaAnalysis) else weak_areas
    areas = [
        area if isinstance(area, WeakArea) else WeakArea.model_validate(area)
        for area in raw_areas
    ]
    priority_order = {"high": 0, "medium": 1, "low": 2}
    areas.sort(key=lambda area: (priority_order[area.priority], area.accuracy, area.topic))
    topic_urls, allowed_urls = _official_resource_map(retrieved_official_resources)
    days: list[StudyPlanDay] = []
    if areas:
        for day_number in range(1, available_days + 1):
            area = areas[(day_number - 1) % len(areas)]
            resources = topic_urls.get(area.topic, []) or [
                url for url in area.source_urls if not allowed_urls or url in allowed_urls
            ]
            days.append(
                StudyPlanDay(
                    day=day_number,
                    topic=area.topic,
                    priority=area.priority,
                    activities=[
                        f"Review the supplied official material for {area.topic}.",
                        "Write a short summary using only the reviewed source.",
                        "Complete a fresh grounded practice check and record errors.",
                    ],
                    resources=list(dict.fromkeys(resources)),
                )
            )
    deterministic = StudyPlan(
        certification=certification, available_days=available_days, days=days
    )
    if not use_openai or not days:
        return deterministic

    response = _resolve_client(client).responses.parse(
        model=_resolve_model(model),
        input=[
            {
                "role": "system",
                "content": (
                    "Improve the wording of this study plan without changing day numbers, "
                    "certification, topics, priorities, or resource URLs. Use no outside resources."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(deterministic.model_dump(mode="json"), ensure_ascii=False),
            },
        ],
        text_format=StudyPlan,
        store=False,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise RuntimeError("OpenAI returned no parsed study plan")
    result = StudyPlan.model_validate(parsed)
    if (
        result.certification != certification
        or result.available_days != available_days
        or [day.day for day in result.days] != [day.day for day in deterministic.days]
        or [day.topic for day in result.days] != [day.topic for day in deterministic.days]
        or [day.priority for day in result.days]
        != [day.priority for day in deterministic.days]
    ):
        raise GroundingValidationError("OpenAI changed deterministic study-plan fields")
    if any(url not in allowed_urls for day in result.days for url in day.resources):
        raise GroundingValidationError("Study plan included an unsupported resource URL")
    return result


def assess_readiness(
    assessment_or_score: AssessmentResult | Mapping[str, Any] | float,
    *,
    ready_threshold: float | None = None,
    nearly_ready_threshold: float | None = None,
    explain_with_openai: bool = False,
    client: Any | None = None,
    model: str | None = None,
) -> ReadinessAssessment:
    """Classify exam readiness from a score using fixed thresholds.

    Deterministic: score >= ready_threshold -> Ready, >= nearly_ready -> Nearly Ready,
    else Needs Improvement (thresholds come from args or env, default 80/60). OpenAI is
    optional and only adds a short explanation; it can never change the score,
    thresholds, or classification.
    """
    load_dotenv()
    ready = ready_threshold if ready_threshold is not None else float(
        os.getenv("READINESS_READY_THRESHOLD", "80")
    )
    nearly = nearly_ready_threshold if nearly_ready_threshold is not None else float(
        os.getenv("READINESS_NEARLY_READY_THRESHOLD", "60")
    )
    if isinstance(assessment_or_score, Mapping):
        score = AssessmentResult.model_validate(assessment_or_score).score_percentage
    elif isinstance(assessment_or_score, AssessmentResult):
        score = assessment_or_score.score_percentage
    else:
        score = float(assessment_or_score)
    deterministic = calculate_readiness(
        score, ready_threshold=ready, nearly_ready_threshold=nearly
    )
    if not explain_with_openai:
        return deterministic
    response = _resolve_client(client).responses.parse(
        model=_resolve_model(model),
        input=[
            {
                "role": "system",
                "content": (
                    "Explain this deterministic readiness result in two concise sentences. "
                    "Do not change its score, thresholds, or classification."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(deterministic.model_dump(mode="json")),
            },
        ],
        text_format=ExplanationOutput,
        store=False,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise RuntimeError("OpenAI returned no parsed readiness explanation")
    return deterministic.model_copy(update={"explanation": parsed.explanation})


__all__ = [
    "analyze_weak_areas",
    "assess_readiness",
    "context_is_sufficient",
    "evaluate_answer",
    "evaluate_assessment",
    "generate_practice_questions",
    "generate_recommendations",
    "generate_study_plan",
    "validate_question_grounding",
]
