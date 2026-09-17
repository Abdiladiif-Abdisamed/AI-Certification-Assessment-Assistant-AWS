"""Certification catalog backed only by repository JSON and derived chunks."""

from fastapi import APIRouter

from ..schemas.api import CertificationSummary
from ..services.certification_service import list_certifications


router = APIRouter(prefix="/certifications", tags=["certifications"])


@router.get("", response_model=list[CertificationSummary])
def certifications() -> list[CertificationSummary]:
    return list_certifications()

