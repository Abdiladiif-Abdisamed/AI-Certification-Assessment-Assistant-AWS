"""Persist practice-question bookmarks for the signed-in learner."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.security import get_current_user
from ..models.entities import Bookmark, User
from ..schemas.api import BookmarkCreate, BookmarkPublic, MessageResponse
from ..services.assessment_service import get_owned_assessment


router = APIRouter(prefix="/bookmarks", tags=["bookmarks"])


def _public(item: Bookmark) -> BookmarkPublic:
    return BookmarkPublic(
        id=item.id,
        question_key=item.question_key,
        certification_id=item.certification_id,
        question=item.question_json,
        note=item.note,
        created_at=item.created_at,
    )


@router.get("", response_model=list[BookmarkPublic])
def list_bookmarks(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[BookmarkPublic]:
    rows = db.scalars(
        select(Bookmark)
        .where(Bookmark.user_id == current_user.id)
        .order_by(Bookmark.created_at.desc())
    ).all()
    return [_public(item) for item in rows]


@router.post("", response_model=BookmarkPublic, status_code=status.HTTP_201_CREATED)
def add_bookmark(
    request: BookmarkCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> BookmarkPublic:
    assessment = get_owned_assessment(db, request.assessment_id, current_user.id)
    if assessment is None:
        raise HTTPException(status_code=404, detail="Assessment was not found.")
    try:
        position = int(request.question_id.removeprefix("q"))
        question = assessment.questions_json[position - 1]
    except (ValueError, IndexError):
        raise HTTPException(status_code=422, detail="Unknown question_id.") from None
    public_question = {
        key: value
        for key, value in question.items()
        if key not in {"correct_answer", "explanation"}
    }
    public_question.update({"question_id": request.question_id, "assessment_id": assessment.id})
    item = Bookmark(
        user_id=current_user.id,
        question_key=f"{assessment.id}:{request.question_id}",
        certification_id=assessment.certification_id,
        question_json=public_question,
        note=request.note.strip(),
    )
    db.add(item)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        existing = db.scalar(
            select(Bookmark).where(
                Bookmark.user_id == current_user.id,
                Bookmark.question_key == f"{assessment.id}:{request.question_id}",
            )
        )
        if existing is None:
            raise
        return _public(existing)
    db.refresh(item)
    return _public(item)


@router.delete("/{bookmark_id}", response_model=MessageResponse)
def delete_bookmark(
    bookmark_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> MessageResponse:
    item = db.scalar(
        select(Bookmark).where(
            Bookmark.id == bookmark_id,
            Bookmark.user_id == current_user.id,
        )
    )
    if item is None:
        raise HTTPException(status_code=404, detail="Bookmark was not found.")
    db.delete(item)
    db.commit()
    return MessageResponse(message="Bookmark removed.")

