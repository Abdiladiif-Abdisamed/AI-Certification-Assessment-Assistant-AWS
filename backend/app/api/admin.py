"""Admin-only platform analytics and exam management views."""

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..core.database import get_db
from ..core.security import get_admin_user
from ..models.entities import Assessment, User
from ..schemas.api import (
    AdminAttemptRow,
    AdminAttemptsResponse,
    AdminCertDetail,
    AdminCertDomain,
    AdminCertUsage,
    AdminExamRow,
    AdminExamsResponse,
    AdminOverview,
    AdminRecentActivity,
    AdminStats,
    AdminUserRow,
    AdminUsersResponse,
    AssessmentResultPublic,
)
from ..services.assessment_service import result_response
from ..services.certification_service import get_certification, list_certifications


router = APIRouter(prefix="/admin", tags=["admin"])

_COMPLETED = Assessment.status == "completed"


@router.get("/overview", response_model=AdminOverview)
def overview(
    db: Session = Depends(get_db), _: User = Depends(get_admin_user)
) -> AdminOverview:
    total_users = db.scalar(select(func.count(User.id))) or 0
    total_assessments, average = db.execute(
        select(func.count(Assessment.id), func.avg(Assessment.score_percentage)).where(
            _COMPLETED
        )
    ).one()

    per_cert_rows = db.execute(
        select(
            Assessment.certification_id,
            Assessment.certification_name,
            func.count(Assessment.id),
            func.avg(Assessment.score_percentage),
        )
        .where(_COMPLETED)
        .group_by(Assessment.certification_id)
        .order_by(func.count(Assessment.id).desc())
    ).all()
    per_certification = [
        AdminCertUsage(
            certification_id=row[0],
            certification_name=row[1],
            attempts=int(row[2] or 0),
            average_score=round(float(row[3] or 0), 2),
        )
        for row in per_cert_rows
    ]

    recent_rows = db.execute(
        select(Assessment, User.email)
        .join(User, User.id == Assessment.user_id)
        .where(_COMPLETED)
        .order_by(Assessment.completed_at.desc())
        .limit(15)
    ).all()
    recent = [
        AdminRecentActivity(
            assessment_id=assessment.id,
            user_email=email,
            certification_name=assessment.certification_name,
            score_percentage=round(float(assessment.score_percentage or 0), 2),
            readiness=(assessment.readiness_json or {}).get(
                "classification", "Not assessed"
            ),
            completed_at=assessment.completed_at,
        )
        for assessment, email in recent_rows
    ]

    return AdminOverview(
        stats=AdminStats(
            total_users=int(total_users),
            total_assessments=int(total_assessments or 0),
            average_score=round(float(average or 0), 2),
        ),
        per_certification=per_certification,
        recent=recent,
    )


@router.get("/users", response_model=AdminUsersResponse)
def users(
    db: Session = Depends(get_db), _: User = Depends(get_admin_user)
) -> AdminUsersResponse:
    stats_rows = db.execute(
        select(
            Assessment.user_id,
            func.count(Assessment.id),
            func.max(Assessment.score_percentage),
            func.avg(Assessment.score_percentage),
        )
        .where(_COMPLETED)
        .group_by(Assessment.user_id)
    ).all()
    stats = {
        row[0]: (int(row[1] or 0), float(row[2] or 0), float(row[3] or 0))
        for row in stats_rows
    }

    rows = []
    for user in db.scalars(select(User).order_by(User.created_at.asc())).all():
        attempts, best, average = stats.get(user.id, (0, 0.0, 0.0))
        rows.append(
            AdminUserRow(
                id=user.id,
                email=user.email,
                full_name=user.full_name,
                is_admin=user.is_admin,
                is_active=user.is_active,
                created_at=user.created_at,
                attempts=attempts,
                best_score=round(best, 2),
                average_score=round(average, 2),
            )
        )
    return AdminUsersResponse(users=rows)


@router.get("/exams", response_model=AdminExamsResponse)
def exams(
    db: Session = Depends(get_db), _: User = Depends(get_admin_user)
) -> AdminExamsResponse:
    attempts_rows = db.execute(
        select(Assessment.certification_id, func.count(Assessment.id)).group_by(
            Assessment.certification_id
        )
    ).all()
    attempts = {row[0]: int(row[1] or 0) for row in attempts_rows}

    exams = [
        AdminExamRow(
            certification_id=cert.certification_id,
            certification_name=cert.certification_name,
            provider=cert.provider,
            domains=cert.domains,
            topics=cert.topics,
            question_generation_ready=cert.question_generation_ready,
            attempts=attempts.get(cert.certification_id, 0),
        )
        for cert in list_certifications()
    ]
    return AdminExamsResponse(exams=exams)


@router.get("/attempts", response_model=AdminAttemptsResponse)
def attempts(
    user_id: int | None = None,
    db: Session = Depends(get_db),
    _: User = Depends(get_admin_user),
) -> AdminAttemptsResponse:
    """Every learner's assessments (optionally filtered to one user)."""
    query = (
        select(Assessment, User.email, User.full_name)
        .join(User, User.id == Assessment.user_id)
        .order_by(Assessment.created_at.desc())
    )
    if user_id is not None:
        query = query.where(Assessment.user_id == user_id)

    rows = []
    for assessment, email, name in db.execute(query).all():
        rows.append(
            AdminAttemptRow(
                assessment_id=assessment.id,
                user_id=assessment.user_id,
                user_email=email,
                user_name=name,
                certification_id=assessment.certification_id,
                certification_name=assessment.certification_name,
                difficulty=assessment.difficulty,
                status=assessment.status,
                score_percentage=assessment.score_percentage,
                readiness=(assessment.readiness_json or {}).get(
                    "classification", "Not assessed"
                ),
                question_count=assessment.question_count,
                created_at=assessment.created_at,
                completed_at=assessment.completed_at,
            )
        )
    return AdminAttemptsResponse(attempts=rows)


@router.get("/assessments/{assessment_id}", response_model=AssessmentResultPublic)
def assessment_detail(
    assessment_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(get_admin_user),
) -> AssessmentResultPublic:
    """Full result of ANY learner's completed assessment (admin can view all)."""
    assessment = db.get(Assessment, assessment_id)
    if assessment is None or assessment.status != "completed":
        raise HTTPException(status_code=404, detail="Completed result was not found.")
    return result_response(assessment)


@router.get("/certifications/{certification_id}", response_model=AdminCertDetail)
def certification_detail(
    certification_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(get_admin_user),
) -> AdminCertDetail:
    """Domain/weight breakdown for one certification (management view)."""
    summary = get_certification(certification_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Certification was not found.")

    settings = get_settings()
    domains: list[AdminCertDomain] = []
    total_topics = 0
    for path in settings.certifications_dir.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        meta = payload.get("certification", {})
        cid = str(meta.get("id") or meta.get("certification_id") or path.stem).strip()
        if cid != certification_id:
            continue
        for domain in payload.get("domains", []) or []:
            if not isinstance(domain, dict):
                continue
            topics = domain.get("topics", []) if isinstance(domain.get("topics", []), list) else []
            total_topics += len(topics)
            domains.append(
                AdminCertDomain(
                    name=str(domain.get("name", "Unnamed domain")),
                    weight=str(domain.get("weight", "")),
                    topics=len(topics),
                )
            )
        break

    attempts = db.scalar(
        select(func.count(Assessment.id)).where(
            Assessment.certification_id == certification_id
        )
    ) or 0

    return AdminCertDetail(
        certification_id=summary.certification_id,
        certification_name=summary.certification_name,
        provider=summary.provider,
        total_domains=len(domains),
        total_topics=total_topics,
        question_generation_ready=summary.question_generation_ready,
        attempts=int(attempts),
        domains=domains,
    )
