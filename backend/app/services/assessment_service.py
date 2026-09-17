"""End-to-end assessment orchestration over the tested RAG/AI engine."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from functools import lru_cache

from openai import APIError
from certification_assistant.ai.engine import (
    AIEngineConfigurationError,
    GroundingValidationError,
    assess_readiness,
    generate_practice_questions,
    generate_recommendations,
    generate_study_plan,
)
from certification_assistant.assessment.evaluation import detect_weak_areas, evaluate_assessment
from certification_assistant.rag.pipeline import FaissRetriever
from certification_assistant.schemas import PracticeQuestion
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..models.entities import Assessment, User
from ..schemas.api import (
    AssessmentCreate,
    AssessmentPublic,
    AssessmentQuestionPublic,
    AssessmentResultPublic,
    AssessmentSubmit,
    HistoryItem,
    QuestionReview,
)
from .certification_service import (
    chunks_for_certification,
    get_certification,
    list_certifications,
)


class AssessmentNotReadyError(ValueError):
    pass


class AssessmentGenerationError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def get_retriever() -> FaissRetriever:
    return FaissRetriever(get_settings().vector_store_dir)


@lru_cache(maxsize=128)
def _cached_retrieval(query: str, certification_id: str, top_k: int) -> tuple:
    """Cache retrieval per query.

    The index and corpus are immutable at runtime and the dashboard sends the same
    default query for a certification every time, so this removes an embedding round
    trip from the critical path on all but the first request.
    """

    return tuple(get_retriever().retrieve_context(query, certification_id, top_k=top_k))


def default_focus_query(certification_name: str) -> str:
    """The retrieval query used when the learner does not supply their own focus.

    Shared with the startup warm-up so both produce the same ``_cached_retrieval``
    key — a warm-up that embedded a slightly different string would cache nothing
    the first real request could use.
    """

    return (
        f"{certification_name} certification knowledge across all domains, "
        f"topics, concepts, services, limitations, and responsible use"
    )


def warm_retrieval_cache() -> None:
    """Load the FAISS index and pre-embed the default query for every ready
    certification, so the first assessment does not pay for either.

    Loading the index alone was not enough: embedding the query is a network round
    trip that measured ~4s cold, and it lands on the first assessment after every
    restart — which under ``uvicorn --reload`` means after every code change.
    """

    get_retriever()
    for certification in list_certifications():
        if not certification.question_generation_ready:
            continue
        corpus = chunks_for_certification(certification.certification_id)
        if not corpus:
            continue
        try:
            _cached_retrieval(
                default_focus_query(certification.certification_name),
                certification.certification_id,
                min(15, len(corpus)),
            )
        except Exception:  # noqa: BLE001 - warm-up is best effort, never fatal
            logging.getLogger(__name__).warning(
                "Could not pre-embed the default query for %s; the first assessment "
                "will pay for it instead.",
                certification.certification_id,
                exc_info=True,
            )


def _question_id(position: int) -> str:
    """Stable per-assessment question id from its 1-based position (e.g. 'q3')."""
    return f"q{position}"


def _public_question(question: PracticeQuestion, position: int) -> AssessmentQuestionPublic:
    """Project an internal question into the client-safe shape — it deliberately
    omits the correct answer and explanation so the exam UI can't leak the key."""
    return AssessmentQuestionPublic(
        question_id=_question_id(position),
        position=position,
        question=question.question,
        options=question.options.model_dump(),
        difficulty=question.difficulty,
        certification=question.certification,
        domain=question.domain,
        topic=question.topic,
        practice_only=True,
    )


def assessment_to_public(assessment: Assessment) -> AssessmentPublic:
    """Convert a stored assessment row into the client payload (answer-key-free
    questions) the exam screen renders."""
    questions = [PracticeQuestion.model_validate(item) for item in assessment.questions_json]
    return AssessmentPublic(
        assessment_id=assessment.id,
        certification_id=assessment.certification_id,
        certification_name=assessment.certification_name,
        difficulty=assessment.difficulty,
        question_count=assessment.question_count,
        available_days=assessment.available_days,
        status=assessment.status,
        questions=[_public_question(question, index) for index, question in enumerate(questions, 1)],
        created_at=assessment.created_at,
    )


def get_owned_assessment(db: Session, assessment_id: str, user_id: int) -> Assessment | None:
    """Fetch an assessment only if it belongs to ``user_id`` (per-user isolation:
    one learner can never read another's assessment)."""
    return db.scalar(
        select(Assessment).where(
            Assessment.id == assessment_id,
            Assessment.user_id == user_id,
        )
    )


def create_assessment(db: Session, user: User, request: AssessmentCreate) -> Assessment:
    """Generate and persist a new practice assessment (the full RAG->generate flow).

    Steps: validate the certification is question-ready -> retrieve grounding context
    from the FAISS index (cached) -> call the AI engine to generate cited, grounded
    MCQs (the batch is split into concurrent shards, each racing parallel candidates,
    so latency tracks the slowest shard rather than the question count) -> if the
    context was too thin, surface a clear "not ready" error -> otherwise store the
    questions and return the new row. No scoring happens here; that is deferred to
    submit.
    """
    certification = get_certification(request.certification_id)
    if certification is None:
        raise AssessmentNotReadyError("Unknown certification selection.")
    if not certification.question_generation_ready:
        raise AssessmentNotReadyError(
            certification.warning
            or "This certification does not yet contain enough detailed official content."
        )

    corpus = chunks_for_certification(request.certification_id)
    query = request.focus_query.strip() or default_focus_query(
        certification.certification_name
    )
    settings = get_settings()
    try:
        retrieved = list(
            _cached_retrieval(query, request.certification_id, min(15, len(corpus)))
        )
        generated = generate_practice_questions(
            certification.certification_name,
            request.difficulty,
            request.question_count,
            retrieved,
            parallel_candidates=settings.generation_candidates,
            shard_size=settings.generation_shard_size,
        )
    except (AIEngineConfigurationError, GroundingValidationError, APIError, RuntimeError) as exc:
        raise AssessmentGenerationError(str(exc)) from exc
    if generated.status != "ok":
        raise AssessmentNotReadyError(generated.reason or "Retrieved context was insufficient.")

    assessment = Assessment(
        user_id=user.id,
        certification_id=request.certification_id,
        certification_name=certification.certification_name,
        difficulty=request.difficulty,
        question_count=len(generated.questions),
        available_days=request.available_days,
        questions_json=[question.model_dump(mode="json") for question in generated.questions],
    )
    db.add(assessment)
    db.commit()
    db.refresh(assessment)
    return assessment


def _result_response(assessment: Assessment) -> AssessmentResultPublic:
    """Assemble the full post-submission result payload (score, weak areas,
    recommendations, study plan, readiness, and per-question review) from the
    fields persisted at submit time."""
    if not assessment.completed_at:
        raise ValueError("Assessment is not completed")
    return AssessmentResultPublic(
        assessment_id=assessment.id,
        certification_id=assessment.certification_id,
        certification_name=assessment.certification_name,
        difficulty=assessment.difficulty,
        status="completed",
        completed_at=assessment.completed_at,
        result=assessment.result_json or {},
        weak_areas=assessment.weak_areas_json or {},
        recommendations=assessment.recommendations_json or {},
        study_plan=assessment.study_plan_json or {},
        readiness=assessment.readiness_json or {},
        question_review=[
            QuestionReview.model_validate(item)
            for item in (assessment.question_review_json or [])
        ],
    )


def submit_assessment(
    db: Session,
    assessment: Assessment,
    submission: AssessmentSubmit,
) -> AssessmentResultPublic:
    """Score a submitted assessment and derive all learning analytics (the full
    submit->results flow).

    Steps: guard against double submission -> map submitted answers back to question
    order (unanswered = wrong) -> DETERMINISTICALLY score with ``evaluate_assessment``
    -> detect weak areas -> build recommendations, a study plan, and a readiness
    verdict (all deterministic here; LLM polish is off) -> build a per-question review
    -> persist everything and mark the assessment completed. The LLM never influences
    the score or pass/fail.
    """
    if assessment.status == "completed":
        return _result_response(assessment)

    questions = [PracticeQuestion.model_validate(item) for item in assessment.questions_json]
    valid_ids = {_question_id(index) for index in range(1, len(questions) + 1)}
    submitted = {item.question_id: item.selected_answer for item in submission.answers}
    unknown = sorted(set(submitted) - valid_ids)
    if unknown:
        raise ValueError("Unknown question_id values: " + ", ".join(unknown))
    answer_list = [submitted.get(_question_id(index), "") for index in range(1, len(questions) + 1)]

    result = evaluate_assessment(questions, answer_list)
    topic_domains = {question.topic: question.domain for question in questions}
    topic_urls: dict[str, list[str]] = {}
    for question in questions:
        topic_urls.setdefault(question.topic, []).extend(question.source_urls)
        topic_urls[question.topic] = list(dict.fromkeys(topic_urls[question.topic]))
    weak_areas = detect_weak_areas(
        result,
        topic_domains=topic_domains,
        topic_source_urls=topic_urls,
    )
    resources = chunks_for_certification(assessment.certification_id)
    recommendations = generate_recommendations(
        weak_areas,
        result.score_percentage,
        assessment.certification_name,
        resources,
        use_openai=False,
    )
    study_plan = generate_study_plan(
        assessment.certification_name,
        weak_areas,
        assessment.available_days,
        retrieved_official_resources=resources,
        use_openai=False,
    )
    settings = get_settings()
    readiness = assess_readiness(
        result,
        ready_threshold=settings.readiness_ready_threshold,
        nearly_ready_threshold=settings.readiness_nearly_ready_threshold,
        explain_with_openai=False,
    )

    review: list[dict] = []
    for position, (question, row) in enumerate(zip(questions, result.question_results, strict=True), 1):
        review.append(
            QuestionReview(
                question_id=_question_id(position),
                position=position,
                question=question.question,
                options=question.options.model_dump(),
                selected_answer=row.selected_answer,
                correct_answer=row.correct_answer,
                is_correct=row.is_correct,
                explanation=question.explanation,
                domain=question.domain,
                topic=question.topic,
                source_chunk_ids=question.source_chunk_ids,
                source_urls=question.source_urls,
                practice_only=True,
            ).model_dump(mode="json")
        )

    assessment.answers_json = submitted
    assessment.result_json = result.model_dump(mode="json")
    assessment.weak_areas_json = weak_areas.model_dump(mode="json")
    assessment.recommendations_json = recommendations.model_dump(mode="json")
    assessment.study_plan_json = study_plan.model_dump(mode="json")
    assessment.readiness_json = readiness.model_dump(mode="json")
    assessment.question_review_json = review
    assessment.status = "completed"
    assessment.score_percentage = result.score_percentage
    assessment.completed_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(assessment)
    return _result_response(assessment)


def latest_completed(db: Session, user_id: int) -> Assessment | None:
    """Return the user's most recently completed assessment (powers the dashboard's
    'latest result' widgets), or None if they have none."""
    return db.scalar(
        select(Assessment)
        .where(Assessment.user_id == user_id, Assessment.status == "completed")
        .order_by(Assessment.completed_at.desc())
        .limit(1)
    )


def history(db: Session, user_id: int) -> list[HistoryItem]:
    """List all of a user's assessments (newest first) as compact history rows for
    the History screen."""
    rows = db.scalars(
        select(Assessment)
        .where(Assessment.user_id == user_id)
        .order_by(Assessment.created_at.desc())
    ).all()
    return [
        HistoryItem(
            assessment_id=item.id,
            certification_id=item.certification_id,
            certification_name=item.certification_name,
            difficulty=item.difficulty,
            question_count=item.question_count,
            status=item.status,
            score_percentage=item.score_percentage,
            readiness=(item.readiness_json or {}).get("classification"),
            created_at=item.created_at,
            completed_at=item.completed_at,
        )
        for item in rows
    ]


def result_response(assessment: Assessment) -> AssessmentResultPublic:
    """Public wrapper around ``_result_response`` (used by the results and admin
    routes to render a completed assessment's full result)."""
    return _result_response(assessment)

