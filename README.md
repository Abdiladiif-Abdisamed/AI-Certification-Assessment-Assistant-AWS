# AI Certification Assistant - RAG and AI Engine

This repository contains a locally integrated certification-assessment MVP:
the independently tested AI foundation, a FastAPI API, SQLite persistence,
JWT authentication, and a React/Vite learner dashboard.

## Project documentation

- [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md) — the examiner-facing
  report: design decisions and their rationale, the five grounding gates, the
  measured latency work, what each test proves, and the known limitations.
- [`docs/AI_Certification_Assistant_Project_Documentation.pdf`](docs/AI_Certification_Assistant_Project_Documentation.pdf)
  — the wider project and five-member collaboration guide.
- [`docs/AI_Certification_Assistant_Project_Documentation.html`](docs/AI_Certification_Assistant_Project_Documentation.html) — editable source of the PDF

The proven flow is:

`official JSON -> cleaning -> topic-aware chunks -> OpenAI embeddings -> certification-scoped FAISS -> grounded Structured Outputs -> deterministic scoring -> weak areas -> recommendations -> study plan -> readiness`

## Current status

- AWS Certified AI Practitioner (`aif-c01`) is the production-candidate corpus.
- AWS has 5 domains, 14 detailed task-level chunks, weights, subtopics, content
  summaries, and official AWS source URLs.
- The vector store is ready with 38 total chunks, OpenAI
  `text-embedding-3-small`, and 1536-dimensional vectors.
- The executed notebook reports 0 retrieval failures, 0 unsupported questions,
  0 invalid questions, and successful live OpenAI retrieval and generation.
- The offline suite currently has 26 passing tests.
- FastAPI now exposes registration/login, certification readiness, assessment
  generation/submission, results, weak areas, recommendations, study plans,
  readiness, history, dashboard summary, and bookmarks.
- The React dashboard is connected to those APIs and no longer uses demo exam
  results or invented certification questions.
- The combined offline suite currently has 30 passing tests; frontend lint and
  production build also pass.
- AI-901, IBM, DeepLearning.AI, and Google are retained, but question generation
  is disabled by the `insufficient_context` guard until their JSON receives the
  same detailed official-source treatment.

This is a quality gate, not a claim that every future model response is correct.
Generated questions must continue to pass schema, source, certification,
all-option, correct-answer, and explanation validation.

## Project layout

```text
ai-certification-assistant/
|-- backend/app/                      FastAPI routes, auth, DB models, services
|-- backend/tests/                    API integration tests
|-- frontend/                         React 19 + Vite integrated learner app
|-- certifications/                  source-of-truth JSON files
|   `-- aws_ai_practitioner.json      detailed AWS production-candidate corpus
|-- data/rag_chunks.json              cleaned topic-aware chunks
|-- notebooks/01_rag_pipeline.ipynb   executed end-to-end evaluation
|-- scripts/
|   |-- create_notebook.py            reproducible notebook generator
|   `-- run_live_pipeline.py          independent live AI smoke test
|-- src/certification_assistant/
|   |-- ingestion/                    loading, cleaning, preparation
|   |-- rag/                          chunking, embeddings, FAISS, retrieval
|   |-- ai/engine.py                  OpenAI generation and grounding gates
|   |-- assessment/                   scoring, weak areas, plans, readiness
|   `-- schemas.py                    strict Pydantic contracts
|-- tests/test_ai_engine.py           AI/RAG offline tests
|-- vector_store/                     global and per-certification FAISS indexes
|-- .env.example                      safe configuration template
|-- pyproject.toml                    installable src-layout package
`-- requirements.txt
```

## 1. Data source and JSON structure

Certification JSON is the runtime source of truth. The pipeline discovers every
`certifications/*.json` file dynamically and never fetches random web content.
It does not contain leaked or real exam questions.

The AWS file is a faithful paraphrase of the official AIF-C01 exam guide, with
one source-linked object per task. It preserves:

- certification ID, name, provider, exam code, and corpus status
- domain name and weight
- topic/task name and subtopics
- detailed scope summary and question focus
- source title, URL, and type

The main official references are the AWS AIF-C01 exam guide, its five content
domain pages, and the official in-scope services page. Unsupported knowledge is
not added to fill gaps.

## 2. Cleaning and normalization

Loading preserves certification and source metadata. Cleaning removes empty
records, exact duplicate content, duplicate source entries, redundant
whitespace, replacement characters, and malformed control characters. It does
not rewrite factual meaning.

The preparation report prints certification, domain, topic, pre-cleaning
document, post-cleaning document, detailed-chunk, and topic-label counts.

## 3. Chunking strategy

One certification topic/task is one semantic unit. Content is split only when a
topic exceeds about 3,500 characters, using paragraph or sentence boundaries and
a small 250-character overlap. Every split retains its certification, provider,
domain, weight, topic, subtopics, source metadata, and stable SHA-256-derived
chunk ID.

This matches exam-guide structure better than arbitrary tiny chunks and reduces
the chance of mixing unrelated objectives.

## 4. Embeddings and FAISS

The embedding adapter uses the OpenAI Python SDK and
`client.embeddings.create(...)`. `OPENAI_EMBEDDING_MODEL` is configurable and
defaults to `text-embedding-3-small`. Vectors are L2-normalized and stored in
FAISS `IndexFlatIP`, so inner product represents cosine similarity.

The store contains a global index for inspection and a dedicated index for each
certification. Model ID, dimensions, chunk metadata, and certification index
mappings are persisted in `vector_store/metadata.json`; embeddings are not
regenerated during normal retrieval.

## 5. Retrieval strategy

```python
from certification_assistant.rag.pipeline import retrieve_context

results = retrieve_context(
    query="prompt engineering techniques",
    certification_id="aif-c01",
    top_k=5,
)
```

The requested certification's FAISS index is selected before query search.
Microsoft content therefore cannot enter an AWS result set. Unknown IDs are
rejected before the query embedding call. Results include chunk ID, text,
domain, topic, source, and cosine similarity score.

## 6. OpenAI integration and question generation

The engine uses the OpenAI Responses API with Pydantic Structured Outputs. The
generation model is configured by `OPENAI_GENERATION_MODEL` and defaults to
`gpt-5-mini`.

`generate_practice_questions(...)` requires certification, difficulty, count,
and retrieved chunks. Its grounding policy states:

> Use only the supplied retrieved context. If the context is insufficient to
> create a reliable question, return insufficient_context instead of inventing
> information.

Accepted questions must satisfy all of these checks:

- exact Pydantic schema and exactly four distinct options
- `practice_only=true`
- requested certification and difficulty
- domain/topic drawn from cited chunks
- only retrieved chunk IDs and URLs
- every option grounded in cited context, including distractors
- correct answer and explanation grounded in cited context
- exact requested count with no duplicate questions

After deterministic checks, a separate closed-book semantic review rejects
unstated definitions, behaviors, cause/effect claims, comparisons, service
capabilities, or best-choice mappings even when their individual words happen to
appear in context. The engine can make up to two regeneration attempts using
validator feedback. If the final output remains unsupported, it raises a
grounding error rather than returning the questions.

### How a batch is scheduled

The questions in one assessment are written by independent shards that run
concurrently, each holding `OPENAI_GENERATION_SHARD_SIZE` questions (default 1),
assigned its own topics, and shown only the retrieved chunks for those topics.
Extra shards are launched alongside the required ones
(`OPENAI_GENERATION_OVERPROVISION`, default 1.0 with a floor of four) and the
first questions to clear every grounding check form the batch; the rest are
abandoned. That matters because shards fail grounding independently, and a
single straggler retrying serially used to cost more wall-clock time than every
other shard put together. Buying that width up front is also what makes the
faster `low` writing effort usable: its extra failures are absorbed in the same
wall-clock window instead of turning into a second round. Anything still missing
is topped up by a second wave, itself over-provisioned, that carries the
rejection reason forward and drops the topic restriction.

Measured on AWS batches with `gpt-5-mini`, from a cold request with the index
already warm:

| Batch | One call for the whole batch | Sharded schedule |
| --- | --- | --- |
| 3 questions | 41-469s | 11-18s |
| 10 questions | 496s | 11-12s |

Validation is identical in both — schema, provenance, lexical grounding, and the
semantic audit all still gate every question. What changed is how the work is
scheduled and how much reasoning budget each call is given.
`OPENAI_GENERATION_EFFORT=medium` buys more deliberation per question at roughly
35s a batch. Below about 10s is not reachable while every question is written
and audited inside the request; that would need questions pre-generated into a
warm pool and served from the database.

Official OpenAI references: [Embeddings guide](https://developers.openai.com/api/docs/guides/embeddings)
and [Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs).

## 7. Deterministic evaluation

`evaluate_answer(...)` and `evaluate_assessment(...)` do not call an LLM. They
calculate total, correct, incorrect, percentage score, per-domain performance,
per-topic performance, and an auditable record for each answer. Missing answers
count as incorrect.

## 8. Weak areas and recommendations

Weak areas come from deterministic domain/topic accuracy first. By default,
topics below 80% require review and topics below 50% are high priority. OpenAI
may optionally explain a result but cannot change the scores.

Recommendations use only detected weak topics and official URLs present in the
retrieved records. Any model-generated topic or URL outside those allowlists is
rejected.

## 9. Study plan

`generate_study_plan(...)` prioritizes the weakest high-priority topics and
cycles them across the available days. Resources are restricted to supplied
official URLs. An optional OpenAI wording pass cannot change the certification,
day numbers, topics, priorities, or resource URLs.

## 10. Readiness

Readiness is deterministic and configurable:

- score >= `READINESS_READY_THRESHOLD` (default 80): `Ready`
- score >= `READINESS_NEARLY_READY_THRESHOLD` (default 60): `Nearly Ready`
- lower score: `Needs Improvement`

OpenAI can explain this result but cannot classify it.

## 11. Setup

Use Python 3.11 or newer. In PowerShell:

```powershell
cd C:\Users\jamaa\Documents\ai-certification-assistant
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[notebook,test]"
Copy-Item .env.example .env
```

Put the secret in `.env`, not `.env.example`:

```dotenv
OPENAI_API_KEY=your_real_key_here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_GENERATION_MODEL=gpt-5-mini
READINESS_READY_THRESHOLD=80
READINESS_NEARLY_READY_THRESHOLD=60
APP_ENVIRONMENT=development
APP_SECRET_KEY=replace-this-with-a-long-local-secret
DATABASE_URL=sqlite:///./data/assistant.db
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

`.env` is ignored by Git. If a real key was ever committed or shared, revoke it
and create a new one.

## 12. Prepare data, build the index, and test

```powershell
cert-rag prepare
cert-rag embed
python -m pytest -q
python scripts\run_live_pipeline.py --questions 3 --days 5
```

The first two commands produce `data/rag_chunks.json`, `vector_store/index.faiss`,
`vector_store/metadata.json`, and `vector_store/by_certification/*.faiss`.
The live script prints retrieval, grounded questions, a demonstration assessment,
weak areas, recommendations, a study plan, and readiness as JSON.

## 13. Run the notebook

Interactive:

```powershell
jupyter notebook notebooks\01_rag_pipeline.ipynb
```

Rebuild and execute it reproducibly:

```powershell
python scripts\create_notebook.py
jupyter nbconvert --to notebook --execute notebooks\01_rag_pipeline.ipynb --output 01_rag_pipeline.ipynb --output-dir notebooks --ExecutePreprocessor.timeout=300
```

The notebook includes 10 cross-certification retrieval tests, pre-search
certification isolation, live OpenAI embeddings/retrieval/generation, JSON schema
validation, all-option grounding, several assessments, weak areas,
recommendations, study plans, readiness boundaries, a concise evaluation report,
and a complete query-to-source-to-evaluation example.

## 14. Run the integrated application

From the project root, start the API in terminal 1:

```powershell
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Start React in terminal 2:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`, register a local account, and choose AWS Certified
AI Practitioner. The API docs are available at `http://127.0.0.1:8000/docs`.
Grounded question generation is a live OpenAI operation and takes roughly half a
minute, because every question is schema-validated and reviewed for support
before the batch is returned.

The default database is `data/assistant.db`. It stores Argon2 password hashes,
never plaintext passwords. Correct answers and explanations remain server-side
until an assessment is submitted.

## 15. API architecture

FastAPI routes call `certification_assistant` as the single AI service layer;
prompts, scoring, retrieval, and readiness logic are not duplicated. Retrieval
selects the certification-specific FAISS index before similarity search. The
backend refuses generation for corpora that lack detailed official content.

Main routes:

- `POST /api/auth/register`, `POST /api/auth/login`, `GET /api/auth/me`
- `GET /api/certifications`, `GET /api/dashboard`
- `POST /api/assessments`, `POST /api/assessments/{id}/submit`
- `GET /api/results/latest`, `GET /api/assessments/history`
- `GET /api/weak-areas/latest`, `/recommendations/latest`, `/study-plan/latest`
- `GET/POST/DELETE /api/bookmarks`

## 16. Verification and deferred work

```powershell
python -m pytest backend/tests tests -q
cd frontend
npm run lint
npm run build
```

Intentionally deferred, as requested:

- full browser end-to-end automation across the live OpenAI flow
- deployment and production security hardening (HTTPS, managed secrets,
  production database/migrations, rate limiting, token rotation/revocation,
  observability, backups, and cloud configuration)
#   A I - C e r t i f i c a t i o n - A s s e s s m e n t - A s s i s t a n t - A W S  
 