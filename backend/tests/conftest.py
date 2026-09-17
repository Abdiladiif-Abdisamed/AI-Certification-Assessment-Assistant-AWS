from __future__ import annotations

import os
import tempfile
from pathlib import Path


TEST_DB = Path(tempfile.gettempdir()) / "ai_certification_assistant_api_tests.db"
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB.as_posix()}"
os.environ["APP_ENVIRONMENT"] = "test"
os.environ["APP_SECRET_KEY"] = "test-secret-that-is-not-used-outside-tests"

import pytest
from fastapi.testclient import TestClient

from backend.app.core.database import Base, engine
from backend.app.main import app


@pytest.fixture(autouse=True)
def clean_database():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/api/auth/register",
        json={
            "email": "learner@example.com",
            "full_name": "Test Learner",
            "password": "correct-horse-battery-staple",
        },
    )
    assert response.status_code == 201
    return {"Authorization": f"Bearer {response.json()['access_token']}"}

