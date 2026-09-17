"""Deterministic scoring, weak-area analysis, and readiness classification."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from ..schemas import (
    AssessmentResult,
    PerformanceBreakdown,
    PracticeQuestion,
    QuestionResult,
    ReadinessAssessment,
    WeakArea,
    WeakAreaAnalysis,
)


def _question_model(question: PracticeQuestion | Mapping[str, Any]) -> PracticeQuestion:
    """Coerce a question (object or plain dict) into a validated ``PracticeQuestion``."""
    if isinstance(question, PracticeQuestion):
        return question
    return PracticeQuestion.model_validate(question)


def _answer_for_index(
    user_answers: Sequence[str] | Mapping[int | str, str], index: int
) -> str:
    """Look up the learner's answer for question ``index``, tolerantly.

    Accepts answers as a list (0-based) or a mapping keyed by 0- or 1-based index
    (int or str). Returns the uppercased letter, or "" when unanswered — an unanswered
    question is thus scored as incorrect, never crashes.
    """
    if isinstance(user_answers, Mapping):
        candidates: tuple[int | str, ...] = (index, index + 1, str(index), str(index + 1))
        for key in candidates:
            if key in user_answers:
                return str(user_answers[key]).strip().upper()
        return ""
    if index >= len(user_answers):
        return ""
    return str(user_answers[index]).strip().upper()


def _breakdowns(rows: list[QuestionResult], field: str) -> dict[str, PerformanceBreakdown]:
    """Aggregate per-group performance (total/correct/incorrect/accuracy) by grouping
    the scored rows on ``field`` — used to build both the per-domain and per-topic
    breakdowns that weak-area analysis later reads."""
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        grouped[getattr(row, field)].append(row.is_correct)
    return {
        label: PerformanceBreakdown(
            total=len(values),
            correct=sum(values),
            incorrect=len(values) - sum(values),
            accuracy=round(100.0 * sum(values) / len(values), 2),
        )
        for label, values in sorted(grouped.items())
    }


def evaluate_assessment(
    questions: Sequence[PracticeQuestion | Mapping[str, Any]],
    user_answers: Sequence[str] | Mapping[int | str, str],
) -> AssessmentResult:
    """Score an MCQ assessment without using an LLM.

    Pure, deterministic scoring — this is the authoritative grade the LLM must never
    override. For each question it compares the learner's selected letter with the
    known correct answer, then returns totals, the percentage, and per-domain /
    per-topic breakdowns used downstream for weak areas and readiness.
    """

    parsed_questions = [_question_model(question) for question in questions]
    rows: list[QuestionResult] = []
    for index, question in enumerate(parsed_questions):
        selected = _answer_for_index(user_answers, index)
        rows.append(
            QuestionResult(
                question_number=index + 1,
                selected_answer=selected,
                correct_answer=question.correct_answer,
                is_correct=selected == question.correct_answer,
                domain=question.domain,
                topic=question.topic,
            )
        )

    total = len(rows)
    correct = sum(row.is_correct for row in rows)
    return AssessmentResult(
        total_questions=total,
        correct=correct,
        incorrect=total - correct,
        score_percentage=round(100.0 * correct / total, 2) if total else 0.0,
        performance_by_domain=_breakdowns(rows, "domain"),
        performance_by_topic=_breakdowns(rows, "topic"),
        question_results=rows,
    )


def detect_weak_areas(
    assessment: AssessmentResult | Mapping[str, Any],
    *,
    topic_domains: Mapping[str, str] | None = None,
    topic_source_urls: Mapping[str, Sequence[str]] | None = None,
    review_threshold: float = 80.0,
    high_priority_below: float = 50.0,
) -> WeakAreaAnalysis:
    """Derive strengths and weaknesses from deterministic assessment aggregates."""

    result = (
        assessment
        if isinstance(assessment, AssessmentResult)
        else AssessmentResult.model_validate(assessment)
    )
    domain_scores = {
        name: values.accuracy for name, values in result.performance_by_domain.items()
    }
    topic_scores = {
        name: values.accuracy for name, values in result.performance_by_topic.items()
    }
    strongest = []
    weakest = []
    if domain_scores:
        maximum = max(domain_scores.values())
        minimum = min(domain_scores.values())
        strongest = sorted(name for name, score in domain_scores.items() if score == maximum)
        weakest = sorted(name for name, score in domain_scores.items() if score == minimum)

    requiring_review = sorted(
        name for name, score in topic_scores.items() if score < review_threshold
    )
    weak_areas: list[WeakArea] = []
    for topic in requiring_review:
        accuracy = topic_scores[topic]
        priority = "high" if accuracy < high_priority_below else "medium"
        domain = ""
        if topic_domains and topic in topic_domains:
            domain = topic_domains[topic]
        else:
            domain = next(
                (row.domain for row in result.question_results if row.topic == topic), ""
            )
        urls = list(dict.fromkeys((topic_source_urls or {}).get(topic, [])))
        weak_areas.append(
            WeakArea(
                domain=domain,
                topic=topic,
                accuracy=accuracy,
                priority=priority,
                source_urls=urls,
            )
        )

    weakest_topics: list[str] = []
    if topic_scores:
        minimum_topic_score = min(topic_scores.values())
        weakest_topics = sorted(
            name for name, score in topic_scores.items() if score == minimum_topic_score
        )

    return WeakAreaAnalysis(
        strongest_domains=strongest,
        weakest_domains=weakest,
        weakest_topics=weakest_topics,
        topics_requiring_review=requiring_review,
        weak_areas=weak_areas,
    )


def calculate_readiness(
    score_percentage: float,
    *,
    ready_threshold: float = 80.0,
    nearly_ready_threshold: float = 60.0,
) -> ReadinessAssessment:
    """Classify readiness using explicit, configurable thresholds."""

    if not 0 <= score_percentage <= 100:
        raise ValueError("score_percentage must be between 0 and 100")
    if not 0 <= nearly_ready_threshold < ready_threshold <= 100:
        raise ValueError(
            "thresholds must satisfy 0 <= nearly_ready_threshold < ready_threshold <= 100"
        )
    if score_percentage >= ready_threshold:
        classification = "Ready"
        explanation = "The assessment score meets the configured ready threshold."
    elif score_percentage >= nearly_ready_threshold:
        classification = "Nearly Ready"
        explanation = "The score meets the nearly-ready threshold but not the ready threshold."
    else:
        classification = "Needs Improvement"
        explanation = "The score is below the configured nearly-ready threshold."
    return ReadinessAssessment(
        score_percentage=round(score_percentage, 2),
        classification=classification,
        ready_threshold=ready_threshold,
        nearly_ready_threshold=nearly_ready_threshold,
        explanation=explanation,
    )
