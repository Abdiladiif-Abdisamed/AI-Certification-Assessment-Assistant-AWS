"""Environment-backed application settings with project-root-safe paths."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseModel):
    app_name: str = "AI Certification Exam Assistant API"
    app_environment: str = Field(default_factory=lambda: os.getenv("APP_ENVIRONMENT", "development"))
    api_prefix: str = "/api"
    secret_key: str = Field(
        default_factory=lambda: os.getenv(
            "APP_SECRET_KEY", "local-development-secret-change-before-production"
        )
    )
    access_token_expire_minutes: int = Field(
        default_factory=lambda: int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440")),
        ge=5,
    )
    database_url: str = Field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", f"sqlite:///{(PROJECT_ROOT / 'data' / 'assistant.db').as_posix()}"
        )
    )
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            origin.strip()
            for origin in os.getenv(
                "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
            ).split(",")
            if origin.strip()
        ]
    )
    certifications_dir: Path = PROJECT_ROOT / "certifications"
    chunks_path: Path = PROJECT_ROOT / "data" / "rag_chunks.json"
    vector_store_dir: Path = PROJECT_ROOT / "vector_store"
    # Independent generations raced inside a single shard. The engine already launches
    # spare shards, which buys the same protection against one bad generation across the
    # whole batch, so racing candidates on top of that only duplicates work: measured on
    # a 10-question batch it was no faster and roughly twice the output tokens.
    generation_candidates: int = Field(
        default_factory=lambda: int(os.getenv("OPENAI_GENERATION_CANDIDATES", "1")),
        ge=1,
        le=5,
    )
    # Questions written per concurrent generation call. A call's latency tracks how much
    # it has to write, so a 10-question exam issued as 10 one-question shards returns in
    # roughly the time of the slowest shard instead of the whole batch. Raise it to make
    # fewer, larger requests when the account's rate limit is the binding constraint.
    generation_shard_size: int = Field(
        default_factory=lambda: int(os.getenv("OPENAI_GENERATION_SHARD_SIZE", "1")),
        ge=1,
        le=20,
    )
    readiness_ready_threshold: float = Field(
        default_factory=lambda: float(os.getenv("READINESS_READY_THRESHOLD", "80"))
    )
    readiness_nearly_ready_threshold: float = Field(
        default_factory=lambda: float(os.getenv("READINESS_NEARLY_READY_THRESHOLD", "60"))
    )

    @field_validator("app_environment")
    @classmethod
    def normalize_environment(cls, value: str) -> str:
        return value.strip().lower()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    if (
        settings.app_environment == "production"
        and settings.secret_key == "local-development-secret-change-before-production"
    ):
        raise RuntimeError("APP_SECRET_KEY must be configured for production")
    return settings

