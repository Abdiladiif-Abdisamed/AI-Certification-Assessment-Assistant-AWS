# FastAPI backend

The API imports the tested `certification_assistant` package for RAG, OpenAI
Structured Outputs, scoring, weak areas, recommendations, study plans, and
readiness. It does not duplicate AI rules in route handlers.

From the project root:

```powershell
python -m pip install -e ".[test]"
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

OpenAPI documentation: `http://127.0.0.1:8000/docs`.

Local defaults use SQLite at `data/assistant.db`, Argon2 password hashing, and
signed bearer tokens. Configure them in the root `.env`; do not place secrets in
`.env.example`.

Run API and AI tests:

```powershell
python -m pytest backend/tests tests -q
```

Production deployment/security hardening is intentionally deferred.

