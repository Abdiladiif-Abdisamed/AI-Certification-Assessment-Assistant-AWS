"""FastAPI entrypoint for the local integrated application."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from .api import admin, assessments, auth, bookmarks, certifications, dashboard
from .core.config import get_settings
from .core.database import create_database_tables
from .services.assessment_service import warm_retrieval_cache


@asynccontextmanager
async def lifespan(_app: FastAPI):
    create_database_tables()
    # Load the FAISS index now rather than inside the first assessment request.
    try:
        await run_in_threadpool(warm_retrieval_cache)
    except Exception:  # noqa: BLE001 - a missing vector store must not block startup
        logging.getLogger(__name__).warning(
            "Vector store could not be preloaded; it will be built on first use.",
            exc_info=True,
        )
    yield


settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    version="0.3.0",
    description="Local API for grounded certification practice and deterministic learning analytics.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix=settings.api_prefix)
app.include_router(certifications.router, prefix=settings.api_prefix)
app.include_router(assessments.router, prefix=settings.api_prefix)
app.include_router(assessments.results_router, prefix=settings.api_prefix)
app.include_router(dashboard.router, prefix=settings.api_prefix)
app.include_router(bookmarks.router, prefix=settings.api_prefix)
app.include_router(admin.router, prefix=settings.api_prefix)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok", "service": "ai-certification-assistant"}

