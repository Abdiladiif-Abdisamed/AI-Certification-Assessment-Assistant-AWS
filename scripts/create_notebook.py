"""Generate the reproducible RAG/AI validation notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "01_rag_pipeline.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


cells = [
    markdown(
        """# AI Certification Assistant: RAG and AI Engine Validation

This notebook validates only the standalone knowledge-to-assessment pipeline. It does not build a frontend, web API, authentication, database, dashboard, or deployment.

The files in `certifications/` are the sole certification knowledge source. No internet material or exam questions are added. Live OpenAI calls are opt-in through `OPENAI_API_KEY`; when the key is absent, deterministic test doubles exercise the vector and structured-output contracts and are reported as offline tests, not as live OpenAI results."""
    ),
    code(
        """from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from dotenv import load_dotenv
from pydantic import ValidationError

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from certification_assistant.ai.engine import (
    analyze_weak_areas,
    assess_readiness,
    evaluate_answer,
    generate_practice_questions,
    generate_recommendations,
    generate_study_plan,
    validate_question_grounding,
)
from certification_assistant.assessment.evaluation import evaluate_assessment
from certification_assistant.rag.pipeline import (
    FaissRetriever,
    build_vector_store,
    chunk_documents,
    clean_documents,
    corpus_quality_report,
    initialize_unbuilt_vector_store,
    load_certification_documents,
    save_chunks,
)
from certification_assistant.schemas import PracticeQuestion, PracticeQuestionBatch, RAGChunk

load_dotenv(ROOT / ".env")
HAS_OPENAI_KEY = bool(os.getenv("OPENAI_API_KEY"))
print(f"Project root: {ROOT}")
print(f"Live OpenAI tests enabled: {HAS_OPENAI_KEY}")"""
    ),
    markdown(
        """## 1. Dynamic ingestion, cleaning, and normalization

Every `*.json` file is discovered dynamically. Topic strings are retained as scope-label knowledge, while richer future topic objects can supply `content`, `description`, `details`, `text`, `knowledge`, `subtopics`, and source metadata. Whitespace/control-character cleanup is conservative and does not paraphrase facts."""
    ),
    code(
        """documents_before, load_stats = load_certification_documents(ROOT / "certifications")
documents, clean_stats = clean_documents(documents_before)
stats = {**load_stats, **clean_stats}

print(f"number of certifications: {stats['number_of_certifications']}")
print(f"number of domains: {stats['number_of_domains']}")
print(f"number of topics: {stats['number_of_topics']}")
print(f"number of documents before cleaning: {stats['documents_before_cleaning']}")
print(f"number after cleaning: {stats['documents_after_cleaning']}")
print(f"empty documents removed: {stats['empty_documents_removed']}")
print(f"duplicate documents removed: {stats['duplicate_documents_removed']}")
print(f"duplicate source records removed: {stats['duplicate_source_records']}")

for certification_id, values in stats["per_certification"].items():
    warning = values["note"] or "No source warning supplied."
    print(f"\\n{certification_id}: {values['documents_before_cleaning']} documents")
    print(f"  {warning}")"""
    ),
    markdown(
        """## 2. Topic-aware chunking

One structured topic is one semantic unit. A topic remains intact unless its supplied content exceeds roughly 3,500 characters. Only then is it split at paragraph/sentence boundaries, with a small 250-character overlap. Certification, provider, domain, weight, topic, subtopics, and source metadata are repeated on every split. This matches certification blueprints better than blind fixed-token fragments and prevents unrelated topics from being mixed."""
    ),
    code(
        """chunks = chunk_documents(documents, max_chars=3500, overlap_chars=250)
save_chunks(chunks, ROOT / "data" / "rag_chunks.json")
quality = corpus_quality_report(chunks)

print(f"Saved {len(chunks)} chunks to data/rag_chunks.json")
print(json.dumps(quality, indent=2))
assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
assert all(chunk.certification_id and chunk.domain and chunk.topic for chunk in chunks)"""
    ),
    markdown(
        """### Corpus sufficiency finding

AWS AIF-C01 now contains detailed, source-linked official exam-guide summaries and is the only certification currently enabled for question generation. The remaining files are intentionally retained as scope-label or incomplete corpora; the generation guard returns `insufficient_context` for them instead of inventing facts."""
    ),
    code(
        """for certification_id in stats["certification_ids"]:
    scoped = [chunk for chunk in chunks if chunk.certification_id == certification_id]
    detailed = sum(chunk.content_detail_level == "detailed" for chunk in scoped)
    print(f"{certification_id}: chunks={len(scoped)}, detailed={detailed}, label_only={len(scoped)-detailed}")"""
    ),
    markdown(
        """## 3. OpenAI embeddings and persisted FAISS

The production path uses `OPENAI_EMBEDDING_MODEL` and writes a global index plus certification-specific indexes. Retrieval chooses the certification index before embedding search. If no key is configured, this cell writes an explicit `not_built` manifest and an empty placeholder index; it never labels fake vectors as OpenAI embeddings."""
    ),
    code(
        """manifest_path = ROOT / "vector_store" / "metadata.json"
cached_manifest = (
    json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest_path.exists() else None
)
expected_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
cache_matches = bool(
    cached_manifest
    and cached_manifest.get("status") == "ready"
    and cached_manifest.get("embedding_provider") == "openai"
    and cached_manifest.get("embedding_model") == expected_model
    and [item["chunk_id"] for item in cached_manifest.get("chunks", [])]
        == [chunk.chunk_id for chunk in chunks]
)
if HAS_OPENAI_KEY and cache_matches:
    production_manifest = cached_manifest
    print(f"Using cached OpenAI vector store: {production_manifest['chunk_count']} vectors")
elif HAS_OPENAI_KEY:
    production_manifest = build_vector_store(chunks, ROOT / "vector_store")
    print(f"Production OpenAI vector store ready: {production_manifest['chunk_count']} vectors")
else:
    production_manifest = initialize_unbuilt_vector_store(chunks, ROOT / "vector_store")
    print("LIVE EMBEDDING TEST SKIPPED:", production_manifest["reason"])

print(json.dumps({key: value for key, value in production_manifest.items() if key not in {"chunks", "certifications"}}, indent=2))"""
    ),
    markdown(
        """## 4. Ten retrieval and isolation tests

The deterministic hash embedder below is test-only. It lets the notebook exercise FAISS persistence and the exact certification-filter path without making or simulating a claim about OpenAI embedding quality."""
    ),
    code(
        """class HashEmbeddingProvider:
    model_name = "test-hash-embedding-v1"

    def __init__(self, dimension=512):
        self.dimension = dimension
        self.query_calls = 0

    def _embed(self, text):
        vector = np.zeros(self.dimension, dtype="float32")
        for raw_token in text.split():
            token = raw_token.strip(".,:;!?()[]").casefold()
            if token:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                vector[int.from_bytes(digest[:4], "little") % self.dimension] += 1.0
        return vector

    def embed_documents(self, texts):
        return np.vstack([self._embed(text) for text in texts])

    def embed_query(self, text):
        self.query_calls += 1
        return self._embed(text)


offline_store = Path(tempfile.mkdtemp(prefix="certification_rag_test_"))
offline_provider = HashEmbeddingProvider()
build_vector_store(chunks, offline_store, embedding_provider=offline_provider)
offline_contract_retriever = FaissRetriever(
    offline_store, embedding_provider=offline_provider
)
if HAS_OPENAI_KEY:
    retriever = FaissRetriever(ROOT / "vector_store")
    retrieval_backend = "live OpenAI embeddings"
else:
    retriever = offline_contract_retriever
    retrieval_backend = "offline hash contract test"
print(f"Offline contract-test store: {offline_store}")
print(f"Retrieval test backend: {retrieval_backend}")"""
    ),
    code(
        """retrieval_cases = [
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
]

retrieval_test_rows = []
for query, certification_id, expected_topic in retrieval_cases:
    results = retriever.retrieve_context(query, certification_id, top_k=5)
    passed = (
        bool(results)
        and all(item.certification_id == certification_id for item in results)
        and expected_topic in {item.topic for item in results}
    )
    retrieval_test_rows.append((query, certification_id, expected_topic, results, passed))
    print(f"\\nQUERY: {query!r} | CERTIFICATION: {certification_id} | PASS: {passed}")
    for item in results[:3]:
        print(f"  {item.similarity_score: .4f} | {item.domain} | {item.topic} | {item.source.source_url}")

assert all(row[-1] for row in retrieval_test_rows)

# Prove the filter is resolved before query embedding, using an observable test double.
calls_before = offline_provider.query_calls
try:
    offline_contract_retriever.retrieve_context(
        "responsible AI", "unknown-certification", top_k=5
    )
except ValueError as exc:
    print("\\nUnknown certification correctly rejected:", exc)
assert offline_provider.query_calls == calls_before"""
    ),
    markdown(
        """## 5. Grounded structured question generation

The contract test uses an actual detailed AWS chunk produced from `aws_ai_practitioner.json`. Every option, the correct answer, and the explanation must pass deterministic citation and lexical support checks. If a live key is available, OpenAI Structured Outputs is tested against retrieved AWS context; incomplete certifications must refuse generation."""
    ),
    code(
        """aws_scope_context = next(
    chunk for chunk in chunks
    if chunk.certification_id == "aif-c01"
    and chunk.topic == "Explain the basic concepts of generative AI"
)

contract_question = PracticeQuestion(
    question="Which supplied term is associated with token-based inference cost?",
    options={
        "A": "tokens",
        "B": "embeddings",
        "C": "vectors",
        "D": "prompts",
    },
    correct_answer="A",
    explanation="The supplied scope connects tokens with token-based inference cost.",
    difficulty="easy",
    certification="AWS Certified AI Practitioner",
    domain="Fundamentals of GenAI",
    topic="Explain the basic concepts of generative AI",
    source_chunk_ids=[aws_scope_context.chunk_id],
    source_urls=[aws_scope_context.source_url],
    practice_only=True,
)


class FakeResponses:
    def __init__(self, result):
        self.result = result
        self.last_kwargs = None

    def parse(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(output_parsed=self.result)


class FakeOpenAI:
    def __init__(self, result):
        self.responses = FakeResponses(result)


fake_client = FakeOpenAI(PracticeQuestionBatch(status="ok", questions=[contract_question]))
generated = generate_practice_questions(
    "AWS Certified AI Practitioner", "easy", 1, [aws_scope_context],
    client=fake_client, model="contract-test-model", use_semantic_grounding_judge=False
)
print(generated.model_dump_json(indent=2))
assert fake_client.responses.last_kwargs["text_format"] is PracticeQuestionBatch
assert "Use only the supplied retrieved context" in json.dumps(fake_client.responses.last_kwargs["input"])

grounding = validate_question_grounding(generated.questions[0], [aws_scope_context])
print("Grounding verdict:", grounding.model_dump())
assert grounding.supported
assert grounding.all_options_grounded

ai901_label_context = [chunk for chunk in chunks if chunk.certification_id == "ai-901"]
refusal = generate_practice_questions(
    "Microsoft Azure AI Fundamentals", "medium", 3, ai901_label_context,
    client=fake_client
)
assert refusal.status == "insufficient_context" and not refusal.questions
print("Current AI-901 corpus result:", refusal.model_dump())

live_generation_test = "SKIPPED - missing OPENAI_API_KEY"
live_generated_count = 0
live_generation_result = None
if HAS_OPENAI_KEY:
    try:
        live_context = retriever.retrieve_context(
            "foundation model prompt engineering techniques and evaluation", "aif-c01", top_k=8
        )
        live_generation_result = generate_practice_questions(
            "AWS Certified AI Practitioner", "medium", 3, live_context
        )
        live_generated_count = len(live_generation_result.questions)
        live_verdicts = [
            validate_question_grounding(question, live_context)
            for question in live_generation_result.questions
        ]
        live_generation_test = "PASSED" if all(v.supported for v in live_verdicts) else "FAILED - unsupported output"
        print(live_generation_result.model_dump_json(indent=2))
        print("Live grounding verdicts:")
        for verdict in live_verdicts:
            print(verdict.model_dump())
    except Exception as exc:
        live_generation_test = f"FAILED - {type(exc).__name__}: {exc}"
print("Live OpenAI generation test:", live_generation_test)"""
    ),
    code(
        """# JSON/schema validity checks.
json_validation_failures = 0
invalid_questions = 0
unsupported_questions = 0
try:
    round_trip = PracticeQuestion.model_validate_json(generated.questions[0].model_dump_json())
    assert round_trip == generated.questions[0]
except (ValidationError, AssertionError):
    json_validation_failures += 1
    invalid_questions += 1
if not grounding.supported:
    unsupported_questions += 1

print("Question schema round-trip: PASS")
print("Correct answer supported:", grounding.correct_answer_supported)
print("All options grounded:", grounding.all_options_grounded)
print("Explanation supported:", grounding.explanation_supported)
print("Practice-only flag:", generated.questions[0].practice_only)
print("Source URL preserved:", generated.questions[0].source_urls == [aws_scope_context.source_url])"""
    ),
    markdown("""## 6. Deterministic assessment, weak areas, recommendations, plan, and readiness"""),
    code(
        """question_two = contract_question.model_copy(update={
    "question": "Which supplied topic belongs to Applications of Foundation Models?",
    "correct_answer": "C",
    "domain": "Applications of Foundation Models",
    "topic": "Choose effective prompt engineering techniques",
    "explanation": "The supplied AWS scope lists prompt engineering in that domain.",
})
question_three = contract_question.model_copy(update={
    "question": "Which supplied topic belongs to the security and governance domain?",
    "correct_answer": "D",
    "domain": "Security, Compliance, and Governance for AI Solutions",
    "topic": "Explain methods to secure AI systems",
    "explanation": "The supplied AWS scope lists AI security in that domain.",
})
assessment_questions = [contract_question, question_two, question_three]
answer_sets = [
    ["A", "C", "D"],
    ["A", "B", "D"],
    ["B", "B", "C"],
]
assessments = [evaluate_assessment(assessment_questions, answers) for answers in answer_sets]
for number, assessment in enumerate(assessments, 1):
    print(f"Assessment {number}: score={assessment.score_percentage}% correct={assessment.correct}/{assessment.total_questions}")
assert [item.score_percentage for item in assessments] == [100.0, 66.67, 0.0]

selected_assessment = assessments[1]
prompt_resource = next(
    chunk for chunk in chunks
    if chunk.certification_id == "aif-c01"
    and chunk.topic == "Choose effective prompt engineering techniques"
)
weak_analysis = analyze_weak_areas(
    selected_assessment,
    topic_source_urls={prompt_resource.topic: [prompt_resource.source_url]},
)
print("\\nWeak areas:", weak_analysis.model_dump_json(indent=2))
recommendations = generate_recommendations(
    weak_analysis,
    selected_assessment.score_percentage,
    "AWS Certified AI Practitioner",
    [prompt_resource],
    use_openai=False,
)
print("\\nRecommendations:", recommendations.model_dump_json(indent=2))

study_plan = generate_study_plan(
    "AWS Certified AI Practitioner",
    weak_analysis,
    3,
    retrieved_official_resources=[prompt_resource],
)
print("\\nStudy plan:", study_plan.model_dump_json(indent=2))

readiness_examples = {score: assess_readiness(score) for score in (85, 70, 55)}
print("\\nReadiness:")
for score, result in readiness_examples.items():
    print(score, result.classification)
assert readiness_examples[85].classification == "Ready"
assert readiness_examples[70].classification == "Nearly Ready"
assert readiness_examples[55].classification == 'Needs Improvement'"""
    ),
    markdown("""## 7. Evaluation report and end-to-end example"""),
    code(
        """retrieval_failures = sum(not row[-1] for row in retrieval_test_rows)
average_retrieved = sum(len(row[3]) for row in retrieval_test_rows) / len(retrieval_test_rows)
live_retrieval_status = (
    "PASSED" if HAS_OPENAI_KEY and retrieval_failures == 0
    else "FAILED" if HAS_OPENAI_KEY
    else "SKIPPED - missing OPENAI_API_KEY"
)

print("RAG TEST RESULTS")
print("----------------")
print(f"Total documents: {stats['documents_after_cleaning']}")
print(f"Total chunks: {len(chunks)}")
print(f"Total certifications: {stats['number_of_certifications']}")
print(f"Average retrieved results: {average_retrieved:.2f}")
print(f"Retrieval failures: {retrieval_failures}")
print(f"Invalid questions: {invalid_questions}")
print(f"Unsupported questions: {unsupported_questions}")
print(f"JSON validation failures: {json_validation_failures}")

print("\\nAI ENGINE TEST RESULTS")
print("----------------------")
print(f"Questions generated: {len(generated.questions) + live_generated_count}")
print(f"Valid questions: {len(generated.questions) - invalid_questions + live_generated_count}")
print(f"Evaluation tests passed: {len(assessments)}")
print(f"Weak-area tests passed: {int(bool(weak_analysis.weak_areas))}")
print(f"Study-plan tests passed: {int(len(study_plan.days) == 3)}")
print(f"Readiness tests passed: {len(readiness_examples)}")
print(f"Live OpenAI embedding test: {'PASSED' if HAS_OPENAI_KEY else 'SKIPPED - missing OPENAI_API_KEY'}")
print(f"Live OpenAI retrieval test: {live_retrieval_status}")
print(f"Live OpenAI generation test: {live_generation_test}")"""
    ),
    code(
        """example_query = "How can token usage affect foundation-model inference cost?"
example_store = Path(tempfile.mkdtemp(prefix="certification_example_"))
example_provider = HashEmbeddingProvider()
build_vector_store([aws_scope_context], example_store, embedding_provider=example_provider)
example_results = FaissRetriever(
    example_store, embedding_provider=example_provider
).retrieve_context(example_query, "aif-c01", top_k=1)
example_evaluation = evaluate_answer(generated.questions[0], "A")

print("USER QUERY")
print(example_query)
print("↓\\nRETRIEVED CONTEXT")
print(example_results[0].text)
print("↓\\nGENERATED QUESTION")
print(generated.questions[0].question)
for label, option in generated.questions[0].options.items():
    print(f"{label}. {option}")
print("↓\\nCORRECT ANSWER")
print(generated.questions[0].correct_answer)
print("↓\\nSOURCE")
print(generated.questions[0].source_urls[0])
print("↓\\nEVALUATION")
print(example_evaluation.model_dump())"""
    ),
    markdown(
        """## Interpretation

The modular pipeline and deterministic logic are operational. The certification boundary is enforced before vector search, unsupported options are rejected, and incomplete certifications safely return `insufficient_context`. AWS AIF-C01 is the current production-candidate corpus. Other certifications must receive the same detailed official-source treatment before question generation is enabled for them."""
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    },
)
NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, NOTEBOOK)
print(f"Wrote {NOTEBOOK}")
