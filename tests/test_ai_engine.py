from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from certification_assistant.ai.engine import (
    GroundingValidationError,
    _overprovision,
    _shard_sizes,
    analyze_weak_areas,
    assess_readiness,
    context_is_sufficient,
    evaluate_answer,
    generate_practice_questions,
    generate_recommendations,
    generate_study_plan,
    validate_question_grounding,
)
from certification_assistant.assessment.evaluation import evaluate_assessment
from certification_assistant.rag.pipeline import (
    FaissRetriever,
    OpenAIEmbeddingProvider,
    build_vector_store,
    corpus_quality_report,
    initialize_unbuilt_vector_store,
    prepare_rag_data,
)
from certification_assistant.schemas import (
    PracticeQuestion,
    PracticeQuestionBatch,
    RAGChunk,
    SemanticGroundingBatch,
    SemanticQuestionReview,
)


ROOT = Path(__file__).resolve().parents[1]


class HashEmbeddingProvider:
    """Deterministic test double; production artifacts never use this provider."""

    model_name = "test-hash-embedding-v1"

    def __init__(self, dimension: int = 512) -> None:
        self.dimension = dimension
        self.query_calls = 0

    def _embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype="float32")
        tokens = [token.strip(".,:;!?()[]").casefold() for token in text.split()]
        for token in tokens:
            if not token:
                continue
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            position = int.from_bytes(digest[:4], "little") % self.dimension
            vector[position] += 1.0
        return vector

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._embed(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        self.query_calls += 1
        return self._embed(text)


@pytest.fixture(scope="module")
def prepared_chunks(tmp_path_factory: pytest.TempPathFactory) -> list[RAGChunk]:
    output = tmp_path_factory.mktemp("chunks") / "rag_chunks.json"
    chunks, stats = prepare_rag_data(ROOT / "certifications", output)
    assert stats["number_of_certifications"] == 5
    assert stats["documents_before_cleaning"] == 38
    assert stats["documents_after_cleaning"] == 38
    assert output.exists()
    assert len(json.loads(output.read_text(encoding="utf-8"))) == len(chunks)
    return chunks


@pytest.fixture(scope="module")
def retriever(
    prepared_chunks: list[RAGChunk], tmp_path_factory: pytest.TempPathFactory
) -> FaissRetriever:
    directory = tmp_path_factory.mktemp("vector_store")
    provider = HashEmbeddingProvider()
    manifest = build_vector_store(
        prepared_chunks, directory, embedding_provider=provider
    )
    assert manifest["status"] == "ready"
    assert (directory / "index.faiss").exists()
    assert (directory / "metadata.json").exists()
    return FaissRetriever(directory, embedding_provider=provider)


def _detailed_aws_chunk() -> RAGChunk:
    return RAGChunk(
        chunk_id="chunk_aws_genai_scope",
        certification_id="aif-c01",
        certification="AWS Certified AI Practitioner",
        provider="AWS",
        domain="Fundamentals of GenAI",
        domain_weight="24%",
        topic="Explain the basic concepts of generative AI",
        subtopics=[],
        text=(
            "Certification: AWS Certified AI Practitioner. Provider: AWS. "
            "Domain: Fundamentals of GenAI. The exact topic name is GenAI concepts. "
            "The supporting terms supplied in this context include foundation models, "
            "tokens, and embeddings."
        ),
        source_url="https://docs.aws.amazon.com/aws-certification/latest/ai-practitioner-01.html",
        source_title="",
        source_type="official",
        source_urls=[
            "https://docs.aws.amazon.com/aws-certification/latest/ai-practitioner-01.html"
        ],
        content_detail_level="detailed",
    )


def _question() -> PracticeQuestion:
    return PracticeQuestion(
        question="What is the exact topic name supplied for this context?",
        options={
            "A": "GenAI concepts",
            "B": "foundation models",
            "C": "tokens",
            "D": "embeddings",
        },
        correct_answer="A",
        explanation=(
            "The context says the exact topic name is GenAI concepts."
        ),
        difficulty="easy",
        certification="AWS Certified AI Practitioner",
        domain="Fundamentals of GenAI",
        topic="Explain the basic concepts of generative AI",
        source_chunk_ids=["chunk_aws_genai_scope"],
        source_urls=[
            "https://docs.aws.amazon.com/aws-certification/latest/ai-practitioner-01.html"
        ],
        practice_only=True,
    )


class FakeResponses:
    def __init__(self, parsed: object | list[object]) -> None:
        self.parsed = parsed if isinstance(parsed, list) else [parsed]
        self.last_kwargs: dict[str, object] = {}
        self.calls: list[dict[str, object]] = []

    def parse(self, **kwargs: object) -> SimpleNamespace:
        self.last_kwargs = kwargs
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self.parsed) - 1)
        return SimpleNamespace(output_parsed=self.parsed[index])


class FakeOpenAI:
    def __init__(self, parsed: object) -> None:
        self.responses = FakeResponses(parsed)


class FakeEmbeddingsEndpoint:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        inputs = kwargs["input"]
        values = inputs if isinstance(inputs, list) else [inputs]
        data = [
            SimpleNamespace(index=index, embedding=[float(len(str(value))), 1.0])
            for index, value in enumerate(values)
        ]
        return SimpleNamespace(data=list(reversed(data)))


class FakeEmbeddingsClient:
    def __init__(self) -> None:
        self.embeddings = FakeEmbeddingsEndpoint()


@pytest.mark.parametrize(
    ("query", "certification_id", "expected_topic"),
    [
        ("responsible AI", "ai-901", "Responsible AI"),
        ("AI model components configurations", "ai-901", "AI model components and configurations"),
        ("generative AI apps agents", "ai-901", "Generative AI apps and agents"),
        ("text and speech", "ai-901", "Text and speech"),
        ("information extraction", "ai-901", "Information extraction"),
        ("AI ML lifecycle", "aif-c01", "Describe the AI/ML development lifecycle"),
        ("capabilities and limitations", "aif-c01", "Understand the capabilities and limitations of GenAI for solving business problems"),
        ("prompt engineering", "aif-c01", "Choose effective prompt engineering techniques"),
        ("responsible AI transparency", "aif-c01", "Explain the development of responsible AI systems"),
        ("AI security governance compliance", "aif-c01", "Explain methods to secure AI systems"),
    ],
)
def test_ten_filtered_retrieval_queries(
    retriever: FaissRetriever,
    query: str,
    certification_id: str,
    expected_topic: str,
) -> None:
    results = retriever.retrieve_context(query, certification_id, top_k=5)
    assert results
    assert all(result.certification_id == certification_id for result in results)
    assert expected_topic in {result.topic for result in results}


def test_unknown_certification_is_rejected_before_query_embedding(
    prepared_chunks: list[RAGChunk], tmp_path: Path
) -> None:
    provider = HashEmbeddingProvider()
    build_vector_store(prepared_chunks, tmp_path, embedding_provider=provider)
    retriever = FaissRetriever(tmp_path, embedding_provider=provider)
    with pytest.raises(ValueError, match="Unknown certification_id"):
        retriever.retrieve_context("responsible AI", "does-not-exist")
    assert provider.query_calls == 0


def test_unbuilt_manifest_is_explicitly_non_queryable(
    prepared_chunks: list[RAGChunk], tmp_path: Path
) -> None:
    manifest = initialize_unbuilt_vector_store(prepared_chunks, tmp_path)
    assert manifest["status"] == "not_built"
    assert (tmp_path / "index.faiss").exists()
    with pytest.raises(RuntimeError, match="not ready"):
        FaissRetriever(tmp_path, embedding_provider=HashEmbeddingProvider())


def test_openai_embedding_adapter_batches_and_preserves_response_order() -> None:
    client = FakeEmbeddingsClient()
    provider = OpenAIEmbeddingProvider(
        model="embedding-test-model", batch_size=2, client=client
    )
    vectors = provider.embed_documents(["a", "four", "seven!!"])
    assert vectors.tolist() == [[1.0, 1.0], [4.0, 1.0], [7.0, 1.0]]
    query = provider.embed_query("query")
    assert query.tolist() == [5.0, 1.0]
    assert len(client.embeddings.calls) == 3
    assert all(call["model"] == "embedding-test-model" for call in client.embeddings.calls)
    assert all(call["encoding_format"] == "float" for call in client.embeddings.calls)


def test_only_aws_corpus_is_question_generation_ready(prepared_chunks: list[RAGChunk]) -> None:
    report = corpus_quality_report(prepared_chunks)
    assert report["ai-901"]["question_generation_ready"] is False
    assert report["aif-c01"]["question_generation_ready"] is True
    assert report["aif-c01"]["detailed_chunks"] == 14
    aws_chunks = [chunk for chunk in prepared_chunks if chunk.certification_id == "aif-c01"]
    assert len(aws_chunks) == 14
    assert len({chunk.domain for chunk in aws_chunks}) == 5
    assert all(chunk.source_url.startswith("https://docs.aws.amazon.com/") for chunk in aws_chunks)
    assert all(chunk.source_type.startswith("official") for chunk in aws_chunks)
    assert "generative-ai-leader" not in report
    assert not context_is_sufficient([chunk for chunk in prepared_chunks if chunk.certification_id == "ai-901"])


def test_question_generation_uses_structured_output_and_grounded_citations() -> None:
    chunk = _detailed_aws_chunk()
    expected = PracticeQuestionBatch(status="ok", questions=[_question()])
    semantic_review = SemanticGroundingBatch(
        all_supported=True,
        reviews=[
            SemanticQuestionReview(
                question_number=1,
                supported=True,
                evidence_chunk_ids=[chunk.chunk_id],
                reason="The relationship is explicit in the supplied context.",
            )
        ],
    )
    client = FakeOpenAI([expected, semantic_review])
    result = generate_practice_questions(
        "AWS Certified AI Practitioner",
        "easy",
        1,
        [chunk],
        client=client,
        model="test-model",
    )
    assert result == expected
    assert client.responses.calls[0]["text_format"] is PracticeQuestionBatch
    assert client.responses.calls[1]["text_format"] is SemanticGroundingBatch
    prompts = json.dumps(client.responses.calls[0]["input"])
    assert "Use only the supplied retrieved context" in prompts
    verdict = validate_question_grounding(result.questions[0], [chunk])
    assert verdict.supported
    assert verdict.all_options_grounded
    assert verdict.unsupported_options == []
    assert result.questions[0].practice_only is True


def test_unsupported_distractor_is_rejected_and_retry_can_recover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Spare shards are the subject of their own tests; switching them off here keeps
    # the call count a statement about the retry path rather than about batch width.
    monkeypatch.setenv("OPENAI_GENERATION_OVERPROVISION", "0")
    chunk = _detailed_aws_chunk()
    unsupported = _question().model_copy(
        update={
            "options": {
                "A": "GenAI concepts",
                "B": "foundation models",
                "C": "tokens",
                "D": "quantum circuit optimization",
            }
        }
    )
    bad_verdict = validate_question_grounding(unsupported, [chunk])
    assert bad_verdict.supported is False
    assert bad_verdict.unsupported_options == ["D"]

    client = FakeOpenAI(
        [
            PracticeQuestionBatch(status="ok", questions=[unsupported]),
            PracticeQuestionBatch(status="ok", questions=[_question()]),
        ]
    )
    result = generate_practice_questions(
        "AWS Certified AI Practitioner",
        "easy",
        1,
        [chunk],
        client=client,
        use_semantic_grounding_judge=False,
    )
    assert result.questions == [_question()]
    assert len(client.responses.calls) == 2
    retry_prompt = json.dumps(client.responses.calls[1]["input"])
    assert "Unsupported answer option(s): D" in retry_prompt

    with pytest.raises(GroundingValidationError, match="failed grounding validation"):
        generate_practice_questions(
            "AWS Certified AI Practitioner",
            "easy",
            1,
            [chunk],
            client=FakeOpenAI(PracticeQuestionBatch(status="ok", questions=[unsupported])),
            max_grounding_attempts=1,
            use_semantic_grounding_judge=False,
        )


def test_semantic_judge_rejects_unstated_relationship_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_GENERATION_OVERPROVISION", "0")
    chunk = _detailed_aws_chunk()
    unsupported_relationship = _question().model_copy(
        update={
            "question": "Which supplied term is guaranteed to eliminate hallucinations?",
            "correct_answer": "B",
            "explanation": (
                "The context lists foundation models as a supporting term for GenAI concepts."
            ),
        }
    )
    rejected = SemanticGroundingBatch(
        all_supported=False,
        reviews=[
            SemanticQuestionReview(
                question_number=1,
                supported=False,
                unsupported_claims=[
                    "The context does not state that foundation models eliminate hallucinations."
                ],
                reason="The tested relationship is not in the context.",
            )
        ],
    )
    approved = SemanticGroundingBatch(
        all_supported=True,
        reviews=[
            SemanticQuestionReview(
                question_number=1,
                supported=True,
                evidence_chunk_ids=[chunk.chunk_id],
                reason="The exact topic-name relationship is stated.",
            )
        ],
    )
    client = FakeOpenAI(
        [
            PracticeQuestionBatch(status="ok", questions=[unsupported_relationship]),
            rejected,
            PracticeQuestionBatch(status="ok", questions=[_question()]),
            approved,
        ]
    )
    result = generate_practice_questions(
        "AWS Certified AI Practitioner", "easy", 1, [chunk], client=client
    )
    assert result.questions == [_question()]
    assert len(client.responses.calls) == 4
    retry_prompt = json.dumps(client.responses.calls[2]["input"])
    assert "does not state that foundation models eliminate hallucinations" in retry_prompt


def test_question_generation_returns_insufficient_context_without_calling_openai(
    prepared_chunks: list[RAGChunk],
) -> None:
    fake = FakeOpenAI(PracticeQuestionBatch(status="ok", questions=[_question()]))
    result = generate_practice_questions(
        "Microsoft Azure AI Fundamentals",
        "medium",
        1,
        [chunk for chunk in prepared_chunks if chunk.certification_id == "ai-901"],
        client=fake,
    )
    assert result.status == "insufficient_context"
    assert result.questions == []
    assert fake.responses.last_kwargs == {}


def test_question_json_schema_rejects_extra_fields_and_duplicate_options() -> None:
    payload = _question().model_dump(mode="json")
    payload["leaked_question"] = True
    with pytest.raises(ValidationError):
        PracticeQuestion.model_validate(payload)
    payload.pop("leaked_question")
    payload["options"]["D"] = payload["options"]["A"]
    with pytest.raises(ValidationError):
        PracticeQuestion.model_validate(payload)


def test_answer_and_assessment_evaluation_are_deterministic() -> None:
    first = _question()
    second = first.model_copy(
        update={
            "question": "Which supplied topic is listed under Applications of Foundation Models?",
            "correct_answer": "C",
            "domain": "Applications of Foundation Models",
            "topic": "prompt engineering",
            "explanation": "The supplied scope lists prompt engineering in that domain.",
        }
    )
    assert evaluate_answer(first, "a").is_correct
    assessment = evaluate_assessment([first, second, second], ["A", "B", "C"])
    assert assessment.total_questions == 3
    assert assessment.correct == 2
    assert assessment.score_percentage == 66.67
    assert assessment.performance_by_topic["prompt engineering"].accuracy == 50.0


def test_weak_areas_recommendations_and_study_plan() -> None:
    first = _question()
    weak_question = first.model_copy(
        update={
            "question": "Which supplied topic is in Applications of Foundation Models?",
            "correct_answer": "C",
            "domain": "Applications of Foundation Models",
            "topic": "prompt engineering",
            "explanation": "The supplied scope lists prompt engineering.",
        }
    )
    assessment = evaluate_assessment([first, weak_question, weak_question], ["A", "B", "B"])
    analysis = analyze_weak_areas(
        assessment,
        topic_source_urls={
            "prompt engineering": [
                "https://docs.aws.amazon.com/aws-certification/latest/ai-practitioner-01.html"
            ]
        },
    )
    assert analysis.weak_areas[0].topic == "prompt engineering"
    assert analysis.weak_areas[0].priority == "high"

    resource = _detailed_aws_chunk().model_copy(
        update={"topic": "prompt engineering", "chunk_id": "chunk_prompt"}
    )
    recommendations = generate_recommendations(
        analysis,
        assessment.score_percentage,
        "AWS Certified AI Practitioner",
        [resource],
        use_openai=False,
    )
    assert recommendations.recommendations[0].what_to_study == "prompt engineering"
    assert recommendations.recommendations[0].recommended_source.startswith("https://docs.aws.amazon.com/")

    plan = generate_study_plan(
        "AWS Certified AI Practitioner",
        analysis,
        3,
        retrieved_official_resources=[resource],
    )
    assert len(plan.days) == 3
    assert [day.day for day in plan.days] == [1, 2, 3]
    assert all(
        resource.source_url in day.resources
        for day in plan.days
    )


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (80, "Ready"),
        (79.99, "Nearly Ready"),
        (60, "Nearly Ready"),
        (59.99, "Needs Improvement"),
    ],
)
def test_readiness_boundaries(score: float, expected: str) -> None:
    assert assess_readiness(score).classification == expected


def test_readiness_threshold_validation() -> None:
    with pytest.raises(ValueError, match="thresholds"):
        assess_readiness(75, ready_threshold=60, nearly_ready_threshold=80)


class ShardAwareFakeResponses:
    """Answers each call with a batch sized to whatever that call actually asked for.

    Shards run concurrently, so this double is locked and keys its reply off the prompt
    rather than off call order the way ``FakeResponses`` does.
    """

    def __init__(self, fail_first: int = 0) -> None:
        self.calls: list[dict[str, object]] = []
        self._lock = threading.Lock()
        self._next_question = 0
        self._failures_left = fail_first

    def parse(self, **kwargs: object) -> SimpleNamespace:
        with self._lock:
            self.calls.append(kwargs)
        prompt = kwargs["input"][1]["content"]
        if kwargs["text_format"] is SemanticGroundingBatch:
            reviewed = json.loads(prompt)["questions"]
            return SimpleNamespace(
                output_parsed=SemanticGroundingBatch(
                    all_supported=True,
                    reviews=[
                        SemanticQuestionReview(
                            question_number=number,
                            supported=True,
                            evidence_chunk_ids=["chunk_aws_genai_scope"],
                            reason="The relationship is explicit in the supplied context.",
                        )
                        for number in range(1, len(reviewed) + 1)
                    ],
                )
            )
        requested = int(re.search(r"Requested question count: (\d+)", prompt).group(1))
        with self._lock:
            start = self._next_question
            self._next_question += requested
            failing = self._failures_left > 0
            if failing:
                self._failures_left -= 1
        update: dict[str, object] = {}
        if failing:
            update["options"] = {
                "A": "GenAI concepts",
                "B": "foundation models",
                "C": "tokens",
                "D": "quantum circuit optimization",
            }
        return SimpleNamespace(
            output_parsed=PracticeQuestionBatch(
                status="ok",
                questions=[
                    _question().model_copy(
                        update={"question": f"Distinct grounded question {index}?", **update}
                    )
                    for index in range(start, start + requested)
                ],
            )
        )


class ShardAwareFakeOpenAI:
    def __init__(self, fail_first: int = 0) -> None:
        self.responses = ShardAwareFakeResponses(fail_first)


def _generation_calls(client: ShardAwareFakeOpenAI) -> list[dict[str, object]]:
    return [
        call for call in client.responses.calls if call["text_format"] is PracticeQuestionBatch
    ]


def test_overprovision_keeps_a_floor_of_spare_shards(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_GENERATION_OVERPROVISION", raising=False)
    # A single shard has nowhere to fan out to, so it retries instead of buying width.
    assert _overprovision(1) == 0
    # The default pairs one spare with every required shard.
    assert _overprovision(3) == 4
    assert _overprovision(10) == 10
    # A ratio small enough to leave a short batch with no slack is lifted to the floor.
    monkeypatch.setenv("OPENAI_GENERATION_OVERPROVISION", "0.3")
    assert _overprovision(3) == 4
    assert _overprovision(10) == 4
    monkeypatch.setenv("OPENAI_GENERATION_OVERPROVISION", "0")
    assert _overprovision(10) == 0


def test_shard_sizes_split_the_batch_evenly() -> None:
    assert _shard_sizes(1, 2) == [1]
    assert _shard_sizes(2, 2) == [2]
    assert _shard_sizes(4, 2) == [2, 2]
    assert _shard_sizes(10, 2) == [2, 2, 2, 2, 2]
    assert _shard_sizes(5, 2) == [2, 2, 1]
    assert _shard_sizes(10, 1) == [1] * 10
    assert sum(_shard_sizes(7, 3)) == 7


def test_generation_shards_a_batch_across_concurrent_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_GENERATION_OVERPROVISION", "0")
    chunk = _detailed_aws_chunk()
    client = ShardAwareFakeOpenAI()
    result = generate_practice_questions(
        "AWS Certified AI Practitioner",
        "easy",
        6,
        [chunk],
        client=client,
        model="test-model",
        shard_size=2,
    )
    assert result.status == "ok"
    assert len(result.questions) == 6
    assert len({question.question for question in result.questions}) == 6
    generation_calls = _generation_calls(client)
    assert len(generation_calls) == 3
    assert all(
        "Requested question count: 2" in call["input"][1]["content"]
        for call in generation_calls
    )


def test_overprovisioned_shard_absorbs_a_failure_without_a_second_wave(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_GENERATION_OVERPROVISION", "1")
    chunk = _detailed_aws_chunk()
    # One shard writes an outside-knowledge distractor; the spare shard launched
    # alongside it supplies the batch instead of a serial retry doing so.
    client = ShardAwareFakeOpenAI(fail_first=1)
    result = generate_practice_questions(
        "AWS Certified AI Practitioner",
        "easy",
        4,
        [chunk],
        client=client,
        model="test-model",
        shard_size=2,
        use_semantic_grounding_judge=False,
    )
    assert result.status == "ok"
    assert len(result.questions) == 4
    assert len({question.question for question in result.questions}) == 4
    assert all(
        "quantum circuit optimization" not in question.options.model_dump().values()
        for question in result.questions
    )
    # Four shards for two shards' worth of questions: two required plus two spares.
    assert len(_generation_calls(client)) <= 4


def test_single_shard_batch_issues_one_generation_call() -> None:
    chunk = _detailed_aws_chunk()
    client = ShardAwareFakeOpenAI()
    result = generate_practice_questions(
        "AWS Certified AI Practitioner",
        "easy",
        2,
        [chunk],
        client=client,
        model="test-model",
        shard_size=4,
    )
    assert len(result.questions) == 2
    assert len(_generation_calls(client)) == 1
