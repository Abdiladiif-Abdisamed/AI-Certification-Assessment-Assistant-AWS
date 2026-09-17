"""Pydantic contracts shared by the RAG pipeline and AI engine."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


AnswerChoice = Literal["A", "B", "C", "D"]
Difficulty = Literal["easy", "medium", "hard"]
Priority = Literal["high", "medium", "low"]


class StrictModel(BaseModel):
    """Base model that rejects fields not included in the public contract."""

    model_config = ConfigDict(extra="forbid")


class SourceReference(StrictModel):
    source_url: str
    source_title: str = ""
    source_type: str = "official"


class RAGMetadata(StrictModel):
    certification_id: str
    certification_name: str
    certification: str
    provider: str
    domain: str
    domain_weight: str = ""
    topic: str
    subtopics: list[str] = Field(default_factory=list)
    content: str
    source_title: str = ""
    source_url: str = ""
    source_type: str = "official"
    source_urls: list[str] = Field(default_factory=list)
    content_detail_level: Literal["detailed", "topic_label"] = "detailed"


class RAGDocument(StrictModel):
    text: str
    metadata: RAGMetadata


class RAGChunk(StrictModel):
    chunk_id: str
    certification_id: str
    certification: str
    provider: str
    domain: str
    domain_weight: str = ""
    topic: str
    subtopics: list[str] = Field(default_factory=list)
    text: str
    source_url: str = ""
    source_title: str = ""
    source_type: str = "official"
    source_urls: list[str] = Field(default_factory=list)
    content_detail_level: Literal["detailed", "topic_label"] = "detailed"


class RetrievalResult(StrictModel):
    chunk_id: str
    certification_id: str
    certification: str
    text: str
    topic: str
    domain: str
    subtopics: list[str] = Field(default_factory=list)
    source: SourceReference
    similarity_score: float
    content_detail_level: Literal["detailed", "topic_label"] = "detailed"


class QuestionOptions(StrictModel):
    A: Annotated[str, Field(min_length=1)]
    B: Annotated[str, Field(min_length=1)]
    C: Annotated[str, Field(min_length=1)]
    D: Annotated[str, Field(min_length=1)]

    @model_validator(mode="after")
    def require_distinct_options(self) -> "QuestionOptions":
        values = [self.A, self.B, self.C, self.D]
        if len({option.strip().casefold() for option in values}) != 4:
            raise ValueError("all four answer options must be distinct")
        return self

    def __getitem__(self, key: AnswerChoice) -> str:
        return getattr(self, key)

    def items(self):
        return self.model_dump().items()


class PracticeQuestion(StrictModel):
    question: Annotated[str, Field(min_length=1)]
    options: QuestionOptions
    correct_answer: AnswerChoice
    explanation: Annotated[str, Field(min_length=1)]
    difficulty: Difficulty
    certification: Annotated[str, Field(min_length=1)]
    domain: Annotated[str, Field(min_length=1)]
    topic: Annotated[str, Field(min_length=1)]
    source_chunk_ids: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)
    source_urls: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)
    practice_only: Literal[True] = True

class PracticeQuestionBatch(StrictModel):
    status: Literal["ok", "insufficient_context"]
    reason: str = ""
    questions: list[PracticeQuestion] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_status_payload(self) -> "PracticeQuestionBatch":
        if self.status == "ok" and not self.questions:
            raise ValueError("status='ok' requires at least one question")
        if self.status == "insufficient_context" and self.questions:
            raise ValueError("insufficient_context must not contain questions")
        return self


class QuestionGroundingVerdict(StrictModel):
    supported: bool
    all_options_grounded: bool
    correct_answer_supported: bool
    explanation_supported: bool
    unsupported_options: list[AnswerChoice] = Field(default_factory=list)
    reason: str


class SemanticQuestionReview(StrictModel):
    question_number: Annotated[int, Field(ge=1)]
    supported: bool
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    reason: str


class SemanticGroundingBatch(StrictModel):
    all_supported: bool
    reviews: list[SemanticQuestionReview]


class ExplanationOutput(StrictModel):
    explanation: str


class AnswerEvaluation(StrictModel):
    is_correct: bool
    selected_answer: str
    correct_answer: AnswerChoice
    explanation: str


class QuestionResult(StrictModel):
    question_number: int
    selected_answer: str
    correct_answer: AnswerChoice
    is_correct: bool
    domain: str
    topic: str


class PerformanceBreakdown(StrictModel):
    total: int
    correct: int
    incorrect: int
    accuracy: float


class AssessmentResult(StrictModel):
    total_questions: int
    correct: int
    incorrect: int
    score_percentage: float
    performance_by_domain: dict[str, PerformanceBreakdown]
    performance_by_topic: dict[str, PerformanceBreakdown]
    question_results: list[QuestionResult]


class WeakArea(StrictModel):
    domain: str
    topic: str
    accuracy: float
    priority: Priority
    source_urls: list[str] = Field(default_factory=list)


class WeakAreaAnalysis(StrictModel):
    strongest_domains: list[str]
    weakest_domains: list[str]
    weakest_topics: list[str]
    topics_requiring_review: list[str]
    weak_areas: list[WeakArea]
    explanation: str = ""


class Recommendation(StrictModel):
    what_to_study: str
    priority: Priority
    recommended_source: str
    reason: str


class RecommendationBatch(StrictModel):
    certification: str
    recommendations: list[Recommendation]


class StudyPlanDay(StrictModel):
    day: int
    topic: str
    priority: Priority
    activities: list[str]
    resources: list[str]


class StudyPlan(StrictModel):
    certification: str
    available_days: int
    days: list[StudyPlanDay]


class ReadinessAssessment(StrictModel):
    score_percentage: float
    classification: Literal["Ready", "Nearly Ready", "Needs Improvement"]
    ready_threshold: float
    nearly_ready_threshold: float
    explanation: str
