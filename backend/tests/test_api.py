from __future__ import annotations

import httpx
from openai import APIConnectionError
from certification_assistant.schemas import PracticeQuestionBatch

from backend.app.services import assessment_service


def test_health_registration_login_and_me(client):
    assert client.get("/health").json()["status"] == "ok"
    created = client.post(
        "/api/auth/register",
        json={
            "email": "USER@example.com",
            "full_name": "API User",
            "password": "strong-password-123",
        },
    )
    assert created.status_code == 201
    assert created.json()["user"]["email"] == "user@example.com"
    token = created.json()["access_token"]
    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert "hashed_password" not in me.text
    login = client.post(
        "/api/auth/login",
        json={"email": "user@example.com", "password": "strong-password-123"},
    )
    assert login.status_code == 200


def test_certification_catalog_reports_only_detailed_corpus_as_ready(client):
    response = client.get("/api/certifications")
    assert response.status_code == 200
    by_id = {item["certification_id"]: item for item in response.json()}
    assert by_id["aif-c01"]["question_generation_ready"] is True
    assert by_id["aif-c01"]["detailed_chunks"] == 14
    assert by_id["ai-901"]["question_generation_ready"] is False
    assert by_id["ai-901"]["warning"]


def test_assessment_flow_hides_answers_then_scores_deterministically(
    client, auth_headers, monkeypatch
):
    questions = PracticeQuestionBatch.model_validate(
        {
            "status": "ok",
            "reason": "",
            "questions": [
                {
                    "question": "Which supplied concept is the correct first practice answer?",
                    "options": {"A": "Alpha", "B": "Beta", "C": "Gamma", "D": "Delta"},
                    "correct_answer": "A",
                    "explanation": "Alpha is explicitly identified by the supplied test context.",
                    "difficulty": "medium",
                    "certification": "AWS Certified AI Practitioner",
                    "domain": "Domain One",
                    "topic": "Topic One",
                    "source_chunk_ids": ["aif-c01-domain-1-task-1-chunk-001"],
                    "source_urls": ["https://docs.aws.amazon.com/test-one"],
                    "practice_only": True,
                },
                {
                    "question": "Which supplied concept is the correct second practice answer?",
                    "options": {"A": "One", "B": "Two", "C": "Three", "D": "Four"},
                    "correct_answer": "B",
                    "explanation": "Two is explicitly identified by the supplied test context.",
                    "difficulty": "medium",
                    "certification": "AWS Certified AI Practitioner",
                    "domain": "Domain Two",
                    "topic": "Topic Two",
                    "source_chunk_ids": ["aif-c01-domain-2-task-1-chunk-001"],
                    "source_urls": ["https://docs.aws.amazon.com/test-two"],
                    "practice_only": True,
                },
            ],
        }
    )

    class FakeRetriever:
        def retrieve_context(self, *_args, **_kwargs):
            return []

    monkeypatch.setattr(assessment_service, "get_retriever", lambda: FakeRetriever())
    monkeypatch.setattr(
        assessment_service,
        "generate_practice_questions",
        lambda *_args, **_kwargs: questions,
    )

    created = client.post(
        "/api/assessments",
        headers=auth_headers,
        json={
            "certification_id": "aif-c01",
            "difficulty": "medium",
            "question_count": 2,
            "available_days": 3,
        },
    )
    assert created.status_code == 201, created.text
    payload = created.json()
    assert payload["question_count"] == 2
    assert "correct_answer" not in payload["questions"][0]
    assert "explanation" not in payload["questions"][0]

    submitted = client.post(
        f"/api/assessments/{payload['assessment_id']}/submit",
        headers=auth_headers,
        json={
            "answers": [
                {"question_id": "q1", "selected_answer": "A"},
                {"question_id": "q2", "selected_answer": "C"},
            ]
        },
    )
    assert submitted.status_code == 200, submitted.text
    result = submitted.json()
    assert result["result"]["score_percentage"] == 50.0
    assert result["readiness"]["classification"] == "Needs Improvement"
    assert result["question_review"][0]["correct_answer"] == "A"
    assert result["question_review"][1]["is_correct"] is False
    assert result["weak_areas"]["weakest_topics"] == ["Topic Two"]
    assert len(result["study_plan"]["days"]) == 3

    history = client.get("/api/assessments/history", headers=auth_headers)
    assert history.status_code == 200
    assert history.json()[0]["score_percentage"] == 50.0

    bookmark = client.post(
        "/api/bookmarks",
        headers=auth_headers,
        json={"assessment_id": payload["assessment_id"], "question_id": "q2"},
    )
    assert bookmark.status_code == 201
    assert "correct_answer" not in bookmark.text


def test_unready_certification_is_rejected_without_calling_openai(client, auth_headers):
    response = client.post(
        "/api/assessments",
        headers=auth_headers,
        json={"certification_id": "ai-901", "difficulty": "easy", "question_count": 2},
    )
    assert response.status_code == 422
    assert "detailed" in response.json()["detail"].lower()


def test_openai_connection_failure_returns_bad_gateway(
    client, auth_headers, monkeypatch
):
    class FakeRetriever:
        def retrieve_context(self, *_args, **_kwargs):
            return []

    monkeypatch.setattr(assessment_service, "get_retriever", lambda: FakeRetriever())
    monkeypatch.setattr(
        assessment_service,
        "generate_practice_questions",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            APIConnectionError(
                message="Connection error.",
                request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
            )
        ),
    )

    response = client.post(
        "/api/assessments",
        headers=auth_headers,
        json={"certification_id": "aif-c01", "difficulty": "easy", "question_count": 1},
    )

    assert response.status_code == 502
    assert "question generation could not" in response.json()["detail"].lower()

