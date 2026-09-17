"""Run the complete AWS AI pipeline without a backend or frontend."""

from __future__ import annotations

import argparse
import json

from certification_assistant.ai.engine import (
    analyze_weak_areas,
    assess_readiness,
    generate_practice_questions,
    generate_recommendations,
    generate_study_plan,
    validate_question_grounding,
)
from certification_assistant.assessment.evaluation import evaluate_assessment
from certification_assistant.rag.pipeline import retrieve_context


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query",
        default="foundation model prompt engineering and evaluation",
    )
    parser.add_argument("--difficulty", choices=("easy", "medium", "hard"), default="medium")
    parser.add_argument("--questions", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--days", type=int, default=5)
    args = parser.parse_args()

    certification_id = "aif-c01"
    certification = "AWS Certified AI Practitioner"
    context = retrieve_context(args.query, certification_id, top_k=args.top_k)
    batch = generate_practice_questions(
        certification,
        args.difficulty,
        args.questions,
        context,
    )
    if batch.status == "insufficient_context":
        print(batch.model_dump_json(indent=2))
        raise SystemExit(2)

    grounding = [
        validate_question_grounding(question, context) for question in batch.questions
    ]
    if not all(verdict.supported for verdict in grounding):
        raise RuntimeError("A generated question failed deterministic grounding validation")

    # A reproducible demonstration assessment: alternating correct and deliberately
    # incorrect answers creates enough signal to exercise weak-area analysis.
    answer_labels = ("A", "B", "C", "D")
    user_answers: list[str] = []
    for index, question in enumerate(batch.questions):
        if index % 2 == 0:
            user_answers.append(question.correct_answer)
        else:
            current = answer_labels.index(question.correct_answer)
            user_answers.append(answer_labels[(current + 1) % len(answer_labels)])

    assessment = evaluate_assessment(batch.questions, user_answers)
    topic_domains = {question.topic: question.domain for question in batch.questions}
    topic_source_urls = {
        question.topic: question.source_urls for question in batch.questions
    }
    weak_areas = analyze_weak_areas(
        assessment,
        topic_domains=topic_domains,
        topic_source_urls=topic_source_urls,
    )
    recommendations = generate_recommendations(
        weak_areas,
        assessment.score_percentage,
        certification,
        context,
        use_openai=False,
    )
    study_plan = generate_study_plan(
        certification,
        weak_areas,
        args.days,
        retrieved_official_resources=context,
    )
    readiness = assess_readiness(assessment)

    print(
        json.dumps(
            {
                "retrieved_context": [item.model_dump(mode="json") for item in context],
                "question_batch": batch.model_dump(mode="json"),
                "grounding": [item.model_dump(mode="json") for item in grounding],
                "demonstration_user_answers": user_answers,
                "assessment": assessment.model_dump(mode="json"),
                "weak_areas": weak_areas.model_dump(mode="json"),
                "recommendations": recommendations.model_dump(mode="json"),
                "study_plan": study_plan.model_dump(mode="json"),
                "readiness": readiness.model_dump(mode="json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
