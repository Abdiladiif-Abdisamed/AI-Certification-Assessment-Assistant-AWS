"""Assessment generation, submission, history, and result routes."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.security import get_current_user
from ..models.entities import User
from ..schemas.api import (
    AssessmentCreate,
    AssessmentPublic,
    AssessmentResultPublic,
    AssessmentSubmit,
    HistoryItem,
)
from ..services.assessment_service import (
    AssessmentGenerationError,
    AssessmentNotReadyError,
    assessment_to_public,
    create_assessment,
    get_owned_assessment,
    history,
    latest_completed,
    result_response,
    submit_assessment,
)


router = APIRouter(prefix="/assessments", tags=["assessments"])
results_router = APIRouter(prefix="/results", tags=["results"])


@router.post("", response_model=AssessmentPublic, status_code=status.HTTP_201_CREATED)
def create(
    request: AssessmentCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AssessmentPublic:
    try:
        assessment = create_assessment(db, current_user, request)
    except AssessmentNotReadyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except AssessmentGenerationError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Question generation could not produce a grounded batch: {exc}",
        ) from exc
    return assessment_to_public(assessment)


@router.get("/history", response_model=list[HistoryItem])
def assessment_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[HistoryItem]:
    return history(db, current_user.id)


@router.get("/{assessment_id}", response_model=AssessmentPublic)
def get_assessment(
    assessment_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AssessmentPublic:
    assessment = get_owned_assessment(db, assessment_id, current_user.id)
    if assessment is None:
        raise HTTPException(status_code=404, detail="Assessment was not found.")
    return assessment_to_public(assessment)


@router.post("/{assessment_id}/submit", response_model=AssessmentResultPublic)
def submit(
    assessment_id: str,
    request: AssessmentSubmit,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AssessmentResultPublic:
    assessment = get_owned_assessment(db, assessment_id, current_user.id)
    if assessment is None:
        raise HTTPException(status_code=404, detail="Assessment was not found.")
    try:
        return submit_assessment(db, assessment, request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@results_router.get("/latest", response_model=AssessmentResultPublic)
def latest_result(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AssessmentResultPublic:
    assessment = latest_completed(db, current_user.id)
    if assessment is None:
        raise HTTPException(status_code=404, detail="No completed assessment is available yet.")
    return result_response(assessment)


@results_router.get("/{assessment_id}", response_model=AssessmentResultPublic)
def get_result(
    assessment_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AssessmentResultPublic:
    assessment = get_owned_assessment(db, assessment_id, current_user.id)
    if assessment is None or assessment.status != "completed":
        raise HTTPException(status_code=404, detail="Completed result was not found.")
    return result_response(assessment)

