"""Learner overview and latest personalized outputs."""

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.security import get_current_user
from ..models.entities import Assessment, User
from ..schemas.api import DashboardSummary
from ..services.assessment_service import latest_completed


router = APIRouter(tags=["dashboard"])


@router.get("/dashboard", response_model=DashboardSummary)
def dashboard(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DashboardSummary:
    completed_filter = (
        Assessment.user_id == current_user.id,
        Assessment.status == "completed",
    )
    count, average, best = db.execute(
        select(
            func.count(Assessment.id),
            func.avg(Assessment.score_percentage),
            func.max(Assessment.score_percentage),
        ).where(*completed_filter)
    ).one()
    latest = latest_completed(db, current_user.id)
    week_ago = datetime.now(timezone.utc) - timedelta(days=7)
    completed_this_week = db.scalar(
        select(func.count(Assessment.id)).where(
            *completed_filter,
            Assessment.completed_at >= week_ago,
        )
    ) or 0
    return DashboardSummary(
        assessments_taken=int(count or 0),
        average_score=round(float(average or 0), 2),
        best_score=round(float(best or 0), 2),
        latest_score=round(float(latest.score_percentage or 0), 2) if latest else 0,
        readiness_classification=(latest.readiness_json or {}).get(
            "classification", "Not assessed"
        ) if latest else "Not assessed",
        completed_this_week=int(completed_this_week),
    )


def _latest_payload(db: Session, user_id: int, field: str) -> dict:
    latest = latest_completed(db, user_id)
    if latest is None:
        raise HTTPException(status_code=404, detail="Complete an assessment first.")
    return getattr(latest, field) or {}


@router.get("/weak-areas/latest")
def latest_weak_areas(
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
) -> dict:
    return _latest_payload(db, current_user.id, "weak_areas_json")


@router.get("/recommendations/latest")
def latest_recommendations(
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
) -> dict:
    return _latest_payload(db, current_user.id, "recommendations_json")


@router.get("/study-plan/latest")
def latest_study_plan(
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
) -> dict:
    return _latest_payload(db, current_user.id, "study_plan_json")


@router.get("/readiness/latest")
def latest_readiness(
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
) -> dict:
    return _latest_payload(db, current_user.id, "readiness_json")

