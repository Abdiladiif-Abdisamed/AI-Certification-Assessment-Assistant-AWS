"""Validated request and response contracts used by the React client."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)


class UserCreate(APIModel):
    email: EmailStr
    full_name: str = Field(min_length=2, max_length=120)
    password: str = Field(min_length=8, max_length=128)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return value.strip().lower()


class LoginRequest(APIModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class UserPublic(APIModel):
    id: int
    email: EmailStr
    full_name: str
    created_at: datetime
    is_admin: bool = False


class AuthResponse(APIModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    user: UserPublic


# --- Admin dashboard -------------------------------------------------------
class AdminStats(APIModel):
    total_users: int
    total_assessments: int
    average_score: float


class AdminCertUsage(APIModel):
    certification_id: str
    certification_name: str
    attempts: int
    average_score: float


class AdminRecentActivity(APIModel):
    assessment_id: str
    user_email: str
    certification_name: str
    score_percentage: float
    readiness: str
    completed_at: datetime | None = None


class AdminOverview(APIModel):
    stats: AdminStats
    per_certification: list[AdminCertUsage]
    recent: list[AdminRecentActivity]


class AdminUserRow(APIModel):
    id: int
    email: EmailStr
    full_name: str
    is_admin: bool
    is_active: bool
    created_at: datetime
    attempts: int
    best_score: float
    average_score: float


class AdminUsersResponse(APIModel):
    users: list[AdminUserRow]


class AdminExamRow(APIModel):
    certification_id: str
    certification_name: str
    provider: str
    domains: int
    topics: int
    question_generation_ready: bool
    attempts: int


class AdminExamsResponse(APIModel):
    exams: list[AdminExamRow]


class AdminAttemptRow(APIModel):
    assessment_id: str
    user_id: int
    user_email: EmailStr
    user_name: str
    certification_id: str
    certification_name: str
    difficulty: str
    status: str
    score_percentage: float | None = None
    readiness: str
    question_count: int
    created_at: datetime
    completed_at: datetime | None = None


class AdminAttemptsResponse(APIModel):
    attempts: list[AdminAttemptRow]


class AdminCertDomain(APIModel):
    name: str
    weight: str = ""
    topics: int


class AdminCertDetail(APIModel):
    certification_id: str
    certification_name: str
    provider: str
    total_domains: int
    total_topics: int
    question_generation_ready: bool
    attempts: int
    domains: list[AdminCertDomain]


class CertificationSummary(APIModel):
    certification_id: str
    certification_name: str
    provider: str
    domains: int
    topics: int
    chunks: int
    detailed_chunks: int
    question_generation_ready: bool
    warning: str = ""


class AssessmentCreate(APIModel):
    certification_id: str
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    question_count: int = Field(default=5, ge=1, le=15)
    available_days: int = Field(default=7, ge=1, le=90)
    focus_query: str = Field(default="", max_length=500)


class QuestionOptionsPublic(APIModel):
    A: str
    B: str
    C: str
    D: str


class AssessmentQuestionPublic(APIModel):
    question_id: str
    position: int
    question: str
    options: QuestionOptionsPublic
    difficulty: str
    certification: str
    domain: str
    topic: str
    practice_only: Literal[True]


class AssessmentPublic(APIModel):
    assessment_id: str
    certification_id: str
    certification_name: str
    difficulty: str
    question_count: int
    available_days: int
    status: str
    questions: list[AssessmentQuestionPublic]
    created_at: datetime


class AnswerSubmission(APIModel):
    question_id: str
    selected_answer: Literal["A", "B", "C", "D"]


class AssessmentSubmit(APIModel):
    answers: list[AnswerSubmission]

    @field_validator("answers")
    @classmethod
    def unique_question_ids(cls, values: list[AnswerSubmission]) -> list[AnswerSubmission]:
        identifiers = [item.question_id for item in values]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Each question_id may be submitted only once")
        return values


class QuestionReview(APIModel):
    question_id: str
    position: int
    question: str
    options: QuestionOptionsPublic
    selected_answer: str
    correct_answer: str
    is_correct: bool
    explanation: str
    domain: str
    topic: str
    source_chunk_ids: list[str]
    source_urls: list[str]
    practice_only: Literal[True]


class AssessmentResultPublic(APIModel):
    assessment_id: str
    certification_id: str
    certification_name: str
    difficulty: str
    status: Literal["completed"]
    completed_at: datetime
    result: dict
    weak_areas: dict
    recommendations: dict
    study_plan: dict
    readiness: dict
    question_review: list[QuestionReview]


class HistoryItem(APIModel):
    assessment_id: str
    certification_id: str
    certification_name: str
    difficulty: str
    question_count: int
    status: str
    score_percentage: float | None
    readiness: str | None
    created_at: datetime
    completed_at: datetime | None


class DashboardSummary(APIModel):
    assessments_taken: int
    average_score: float
    best_score: float
    latest_score: float
    readiness_classification: str
    completed_this_week: int


class BookmarkCreate(APIModel):
    assessment_id: str
    question_id: str
    note: str = Field(default="", max_length=1000)


class BookmarkPublic(APIModel):
    id: int
    question_key: str
    certification_id: str
    question: dict
    note: str
    created_at: datetime


class MessageResponse(APIModel):
    message: str

