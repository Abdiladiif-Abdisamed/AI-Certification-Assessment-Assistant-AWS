# AI Certification Assistant — Technical Report

**Version 0.3.0 · Reviewed against the code at commit time of writing**

This report is written for an examiner reading the source alongside it. Every
claim below names the file that implements it, and the verification section
gives the commands that reproduce each result.

---

## 1. Summary

The system generates certification practice questions from official exam-guide
content, scores a learner's attempt, and reports where they are weak and whether
they are ready to sit the real exam.

The design problem is not "call an LLM to write questions" — that is trivial and
unreliable. The problem is producing questions that are **provably traceable to
official source material**, and being able to demonstrate that a question the
model invented from its own training data will be rejected before a learner ever
sees it. Most of the engineering in this repository exists to enforce that.

Two properties follow from that decision and shape everything else:

- **The language model never decides anything that affects a learner's result.**
  Scoring, weak-area detection, and readiness classification are ordinary
  deterministic Python. The model writes question text and optional prose; it
  cannot change a score.
- **A question must pass five independent gates before it is stored.** Four are
  local and free; the fifth is a separate model call acting as an auditor. If a
  question fails, it is discarded and regenerated, and if the corpus cannot
  support grounded questions at all, the API returns a refusal rather than
  guessing.

---

## 2. Scope and non-goals

### In scope

- Ingesting certification exam-guide content from versioned JSON files
- A retrieval-augmented pipeline with per-certification vector isolation
- Grounded multiple-choice generation with provenance enforcement
- Deterministic scoring, weak-area analysis, study planning, and readiness
- A FastAPI backend with JWT auth and SQLite persistence
- A React learner dashboard covering the full attempt-to-result flow

### Deliberately not in scope

- **Real or leaked exam questions.** The corpus is a paraphrase of publicly
  published exam guides. Every generated question carries `practice_only=True`
  as a schema-enforced literal (`schemas.py:110`), so a question object that
  claims otherwise cannot be constructed.
- **Production deployment hardening.** No HTTPS termination, managed secrets,
  database migrations, rate limiting, token revocation, or observability. The
  system is an MVP intended to run locally.
- **Certifications without detailed source content.** Three of the four corpora
  are present but generation is refused for them. Section 4 explains why this is
  a feature rather than an omission.

---

## 3. System overview

```
certifications/*.json          official exam-guide content (source of truth)
        |
        v  ingestion/loader.py, cleaning.py, preparation.py
   cleaned documents           duplicates, control characters, empty records removed
        |
        v  rag/chunking.py
   topic-aware chunks          one exam task = one chunk; SHA-256 chunk IDs
        |
        v  rag/embeddings.py, vector_store.py
   FAISS indexes               one global index + one index per certification
        |
        v  rag/retriever.py
   retrieved context           certification chosen BEFORE the similarity search
        |
        v  ai/engine.py
   generated questions         Structured Outputs + 5 acceptance gates
        |
        v  assessment/*.py
   score, weak areas,          deterministic; no model involvement
   study plan, readiness
```

The request path through the application is:

`React → FastAPI route → service layer → certification_assistant → OpenAI/FAISS`

The backend contains no prompts, no scoring rules, and no retrieval logic of its
own. `backend/app/services/` is a thin orchestration layer over the
`certification_assistant` package, which is independently testable without
FastAPI, a database, or a browser. This is why the AI test suite
(`tests/test_ai_engine.py`) can run offline against injected fake clients.

---

## 4. Knowledge base

### Source of truth

`certifications/*.json` is the runtime source of truth. The pipeline discovers
files dynamically and never fetches arbitrary web content at runtime. The AWS
file is a faithful paraphrase of the published AIF-C01 exam guide, structured as
one object per exam task, each carrying its domain, weight, subtopics, scope
summary, and the official AWS URL it came from.

### Corpus status

| Certification | Chunks | Detailed chunks | Generation |
| --- | --- | --- | --- |
| AWS Certified AI Practitioner (`aif-c01`) | 14 | 14 | Enabled |
| Microsoft Azure AI Fundamentals (`ai-901`) | 7 | 0 | Refused |
| IBM AI Engineering Professional Certificate | 10 | 0 | Refused |
| DeepLearning.AI programs | 7 | 0 | Refused |

Total: 38 chunks. AWS covers 5 domains.

### Why three corpora are refused rather than removed

`context_is_sufficient()` (`ai/engine.py`) requires at least one retrieved chunk
that is marked `detailed`, contains 120 or more characters of explanatory text,
**and** carries an official source URL. The three non-AWS files currently hold
topic labels — the names of exam objectives — without the scope prose behind
them. A model given only a list of topic names can still write a plausible
question, but that question would be grounded in the model's training data, not
in the cited source. The pipeline therefore returns `insufficient_context` and
the API answers HTTP 422 with a readable reason.

This is the most important behaviour in the project to evaluate: the system was
built to prefer refusing over inventing, and the refusal path is covered by
tests (`test_only_aws_corpus_is_question_generation_ready`,
`test_question_generation_returns_insufficient_context_without_calling_openai`,
`test_unready_certification_is_rejected_without_calling_openai`). The last two
assert that **no OpenAI call is made at all** on the refusal path.

---

## 5. Retrieval

### Chunking

One certification topic or task is one semantic unit. A topic is split only when
it exceeds 3,500 characters, on paragraph or sentence boundaries, with a
250-character overlap (`rag/pipeline.py`, `max_chars` and `overlap_chars`). Every
chunk keeps its certification, provider, domain, weight, topic, subtopics, source
metadata, and a stable SHA-256-derived `chunk_id`.

The alternative — fixed-size chunks of a few hundred tokens — was rejected
because exam guides are already structured by objective. Arbitrary splits mix
unrelated objectives into one chunk, which makes the domain/topic provenance
check in Section 6 meaningless.

### Embeddings and index

`text-embedding-3-small` (configurable via `OPENAI_EMBEDDING_MODEL`), 1536
dimensions. Vectors are L2-normalised and stored in a FAISS `IndexFlatIP`, so
the inner product is exactly cosine similarity. `IndexFlatIP` is an exhaustive
index: at 38 chunks, an approximate index would add failure modes and complexity
for no measurable speed benefit.

`vector_store/metadata.json` persists the model ID, dimensions, chunk metadata,
and the per-certification index mapping. Embeddings are not regenerated during
normal retrieval.

### Certification isolation

The system stores a **separate FAISS index per certification** in addition to
the global one, and selects the index before running the query. Cross-corpus
leakage is therefore structurally impossible rather than filtered out
afterwards — Microsoft content cannot appear in an AWS result set because it was
never in the searched index. Unknown certification IDs are rejected before the
query embedding call is made, so a bad ID costs nothing.

Evidence: `test_ten_filtered_retrieval_queries` (10 parametrised cases across
two certifications) and `test_unknown_certification_is_rejected_before_query_embedding`.

---

## 6. Question generation and the acceptance gates

Generation uses the OpenAI Responses API with Pydantic Structured Outputs.
The default model is `gpt-5-mini` (`OPENAI_GENERATION_MODEL`).

The system prompt restricts the model to the supplied context and instructs it
to return `insufficient_context` rather than invent. That instruction is not
trusted. Every returned question passes through the following gates, in order,
in `ai/engine.py`.

### Gate 1 — Schema

`PracticeQuestion` is a `StrictModel` (`extra="forbid"`). Unknown fields are a
validation error, not ignored input. `QuestionOptions` enforces exactly four
distinct options, `correct_answer` is constrained to A–D, `source_chunk_ids` and
`source_urls` must each be non-empty, and `practice_only` is the literal `True`.

A model response that adds a field — for instance one claiming to be a real exam
item — fails to parse. `test_question_json_schema_rejects_extra_fields_and_duplicate_options`
covers both the extra-field and duplicate-option cases.

### Gate 2 — Provenance

Implemented in `validate_question_grounding()`. Every cited `chunk_id` must
exist in the retrieved set; every cited `source_url` must belong to one of the
chunks the question itself cited; and the question's `domain` and `topic` must
match the chunks it cited. A question cannot cite a chunk about prompt
engineering and then claim to be testing responsible AI.

### Gate 3 — Lexical grounding of every option

Each of the four options must either appear verbatim in the cited context, or
share at least 80% of its meaningful tokens with it (stopwords and tokens of
three characters or fewer are excluded). This applies to the **distractors as
well as the correct answer** — a plausible-sounding wrong answer drawn from the
model's own knowledge is exactly the failure this gate exists to catch. The
explanation must reach a 30% token-overlap threshold; it is prose rather than a
copied phrase, so the bar is lower.

`test_unsupported_distractor_is_rejected_and_retry_can_recover` constructs a
question whose only defect is a fourth option reading "quantum circuit
optimization" — absent from the context — and asserts both that it is rejected
and that the rejection reason is fed back into the regeneration prompt.

### Gate 4 — Semantic audit

Lexical overlap cannot catch a question that reuses the right words to assert a
relationship the source never states. A second model call, prompted as a
closed-book provenance auditor, reviews each question against the cited context
and must return per-question evidence chunk IDs. It rejects unstated
definitions, cause-and-effect claims, comparisons, capability claims, and
best-choice mappings.

The auditor's own output is validated too: it must return one ordered result per
question, may only cite chunk IDs that were in the context, may not approve a
question without naming evidence, and its aggregate `all_supported` flag must
agree with its individual verdicts. An auditor that contradicts itself is
treated as a failed review.

`test_semantic_judge_rejects_unstated_relationship_and_retries` uses a question
asking which supplied term "is guaranteed to eliminate hallucinations" — every
word is in the context, the relationship is not — and asserts the auditor
rejects it and that the rejection text reaches the retry prompt.

### Gate 5 — Batch integrity

The batch must contain exactly the requested number of questions with no
duplicate question text, and every question must carry the requested
certification and difficulty.

### On failure

A failed batch is regenerated with the specific validator feedback appended to
the prompt, up to the configured attempt limit. If the output is still
unsupported, the engine raises `GroundingValidationError` and the API returns
HTTP 502 rather than degraded questions. **There is no path by which a question
that failed a gate reaches the learner.**

---

## 7. Generation scheduling and latency

This section documents a measured performance problem and its fix, because the
naive implementation was unusable and the reasoning is worth examining.

### The original problem

A single API call was asked to write all N questions, then a second call audited
them. Measured on a 10-question AWS batch with `gpt-5-mini`, this took **496
seconds**. Profiling every OpenAI round trip showed why: generation latency
scales with how much the model has to write, and when one batch failed a gate,
the *entire* batch was rewritten serially.

A profile of a 10-question request showed a single unlucky shard retrying three
times in sequence, consuming 110 seconds of a 150-second request — more than
every other unit of work combined.

### The fix

Three changes, all in `ai/engine.py`:

1. **Sharding.** Each question is written by its own concurrent call
   (`OPENAI_GENERATION_SHARD_SIZE`, default 1). Wall-clock time now tracks the
   slowest shard rather than the total question count.
2. **Per-shard context narrowing.** Topics are dealt round-robin across shards,
   and each shard is shown only the chunks for its own topics. Prompt input fell
   from ~25,000 characters to ~3,200 per call. This also keeps shards from
   writing the same question.
3. **Over-provisioning instead of retrying.** Spare shards are launched
   alongside the required ones (`OPENAI_GENERATION_OVERPROVISION`, default 1.0
   with a floor of four spares). The first questions to clear every gate form
   the batch; the rest are abandoned. A shard that fails no longer serialises a
   retry onto the end of the request — it is simply not one of the winners.

Buying width up front is also what makes a lower reasoning effort usable: `low`
fails gates more often than `medium`, but those extra failures are absorbed
inside the same wall-clock window.

### Measured results

`gpt-5-mini`, warm index, medians over repeated runs:

| Batch size | Single call for the whole batch | Sharded schedule |
| --- | --- | --- |
| 3 questions | 41–469 s | 11–18 s |
| 10 questions | 496 s | 11–12 s |

The wide range on the original 3-question figure is the point: the old schedule
was not merely slow, it was unpredictable, because a single gate failure changed
the response time by an order of magnitude.

### What was and was not traded away

Every gate in Section 6 runs unchanged. What changed is scheduling and the
reasoning budget given to each call.

The auditor's effort was lowered from `medium` to `minimal`, which saves several
seconds per request. This was **verified rather than assumed**: a known
unsupported question and a known supported control were each put to the auditor
five times at every effort level. `minimal`, `low`, and `medium` all rejected
the unsupported question 5/5 and approved the control 5/5. The extra reasoning
budget was buying latency, not accuracy.

Below roughly 10 seconds is not reachable while every question is both written
and audited inside the request — the floor is one generation round trip plus one
audit round trip. Going lower would require pre-generating validated questions
into a warm pool and serving them from the database.

---

## 8. Scoring and learning analytics

Everything in this section is deterministic Python in
`src/certification_assistant/assessment/`. No model call can alter any of it.

**Scoring** (`evaluate_assessment`) compares each submitted answer against the
stored answer key and produces totals, a percentage, per-domain accuracy,
per-topic accuracy, and an auditable per-question record. An unanswered question
counts as incorrect.

**Weak areas** (`detect_weak_areas`) marks any topic below 80% accuracy as
requiring review, and any topic below 50% as high priority. Both thresholds are
parameters, not literals.

**Recommendations and study plan** may optionally use the model for wording, but
operate under allowlists: a topic the learner was not actually weak in, or a URL
that was not in the retrieved official sources, is rejected. The wording pass
cannot change the certification, day numbers, topics, priorities, or URLs.

**Readiness** (`calculate_readiness`) is a threshold comparison:
`≥ READINESS_READY_THRESHOLD` (default 80) is *Ready*,
`≥ READINESS_NEARLY_READY_THRESHOLD` (default 60) is *Nearly Ready*, otherwise
*Needs Improvement*. The thresholds are validated on entry, so a misconfiguration
raises rather than silently producing nonsense classifications.

### Answer-key confidentiality

`AssessmentQuestionPublic` deliberately omits `correct_answer` and `explanation`
(`assessment_service.py:_public_question`). While an assessment is in progress
the answer key exists only in the database; it is not sent to the browser in any
form, so it cannot be read out of network traffic or developer tools. Both are
returned only in the post-submission review payload.

`test_assessment_flow_hides_answers_then_scores_deterministically` asserts this
end to end through the HTTP API.

---

## 9. Application layer

### API

FastAPI, all routes under `/api`. Authentication is a signed JWT bearer token
(HS256), and assessments are fetched through an owner-scoped query
(`get_owned_assessment`), so one learner cannot read another's attempt even with
a valid token and a guessed UUID.

| Area | Routes |
| --- | --- |
| Auth | `POST /auth/register`, `POST /auth/login`, `GET /auth/me` |
| Catalogue | `GET /certifications`, `GET /dashboard` |
| Assessments | `POST /assessments`, `GET /assessments/{id}`, `POST /assessments/{id}/submit`, `GET /assessments/history` |
| Results | `GET /results/latest`, `GET /results/{id}` |
| Analytics | `GET /weak-areas/latest`, `/recommendations/latest`, `/study-plan/latest`, `/readiness/latest` |
| Bookmarks | `GET`, `POST`, `DELETE /bookmarks/{id}` |
| Admin | `GET /admin/overview`, `/users`, `/exams`, `/attempts`, `/assessments/{id}`, `/certifications/{id}` |

Interactive documentation is generated at `/docs`.

### Database

SQLAlchemy 2.0 over SQLite (`data/assistant.db`). Three tables:

- **`users`** — email (unique, indexed), full name, password hash, active and
  admin flags. Passwords are hashed with `pwdlib`'s recommended hasher (Argon2,
  with a bcrypt fallback for Windows environments where the native Argon2
  bindings are blocked). Plaintext passwords are never stored.
- **`assessments`** — UUID primary key, owner FK with cascade delete, and the
  questions, answers, result, weak areas, recommendations, study plan, readiness
  and per-question review each stored as JSON columns. Storing the derived
  analytics rather than recomputing them means a historical result cannot
  silently change if a threshold is later reconfigured.
- **`bookmarks`** — a unique constraint on `(user_id, question_key)` makes
  double-bookmarking a database-level impossibility rather than an application
  check.

### Frontend

React 19 with Vite 8, no UI framework dependency beyond `lucide-react` icons.
Eleven screens plus a shared sidebar: authentication, dashboard, exam, results,
weak areas, study plan, history, bookmarks, settings, help, admin. The API client
(`src/services/api.js`) centralises the base URL, bearer token injection, error
normalisation, and per-request timeouts.

---

## 10. Verification

### Reproducing the results

```bash
python -m pytest tests backend/tests -q
```

```bash
cd frontend && npm run lint && npm run build
```

```bash
python scripts/run_live_pipeline.py --questions 3 --days 5
```

### Current state

- **36 automated tests pass** across the AI/RAG suite and the API integration
  suite.
- **Frontend lint (oxlint) passes with no findings; the production build
  succeeds** (1,830 modules, 262 kB JS / 78 kB gzipped).
- The live pipeline script exercises real retrieval, generation, scoring, weak
  areas, recommendations, study plan, and readiness against the OpenAI API and
  prints each stage as JSON.

### What the tests actually prove

The suite is organised around the guarantees, not around line coverage:

| Guarantee | Tests |
| --- | --- |
| Retrieval is certification-isolated | `test_ten_filtered_retrieval_queries` (10 cases), `test_unknown_certification_is_rejected_before_query_embedding` |
| Thin corpora are refused without calling OpenAI | `test_only_aws_corpus_is_question_generation_ready`, `test_question_generation_returns_insufficient_context_without_calling_openai`, `test_unready_certification_is_rejected_without_calling_openai` |
| The schema rejects malformed and extra-field output | `test_question_json_schema_rejects_extra_fields_and_duplicate_options` |
| Ungrounded distractors are rejected and retried with feedback | `test_unsupported_distractor_is_rejected_and_retry_can_recover` |
| Unstated relationships are caught by the auditor | `test_semantic_judge_rejects_unstated_relationship_and_retries` |
| Structured Outputs and citations are used as specified | `test_question_generation_uses_structured_output_and_grounded_citations` |
| Concurrent scheduling produces correct, unique batches | `test_generation_shards_a_batch_across_concurrent_calls`, `test_overprovisioned_shard_absorbs_a_failure_without_a_second_wave`, `test_single_shard_batch_issues_one_generation_call`, `test_shard_sizes_split_the_batch_evenly`, `test_overprovision_keeps_a_floor_of_spare_shards` |
| Scoring is deterministic and answers stay hidden until submission | `test_answer_and_assessment_evaluation_are_deterministic`, `test_assessment_flow_hides_answers_then_scores_deterministically` |
| Readiness boundaries are exact | `test_readiness_boundaries` (4 boundary cases), `test_readiness_threshold_validation` |
| Analytics derive only from detected weakness | `test_weak_areas_recommendations_and_study_plan` |
| Auth and the catalogue behave | `test_health_registration_login_and_me`, `test_certification_catalog_reports_only_detailed_corpus_as_ready` |
| An OpenAI outage surfaces as 502, not a stack trace | `test_openai_connection_failure_returns_bad_gateway` |

The AI tests inject fake OpenAI clients, so the suite runs offline,
deterministically, and without API cost. The vector-store tests build a real
FAISS index using a deterministic hash-based embedding provider — a test double
that is never used in production paths.

---

## 11. Known limitations

These are stated plainly because an examiner will find them anyway.

1. **One production-ready corpus.** Only AWS AIF-C01 has the detailed content
   the gates require. The other three are wired up but refused. Extending the
   system is a content task, not a code task.
2. **No end-to-end browser automation.** The full live flow has been exercised
   manually and through the live pipeline script, but there is no automated
   Playwright/Cypress run across the real OpenAI path.
3. **Not deployment-hardened.** No HTTPS, managed secrets, migrations, rate
   limiting, token revocation, observability, or backups. SQLite and a
   development secret key are appropriate for local evaluation only.
4. **Generation cost scales with over-provisioning.** The default doubles the
   number of generation calls to buy latency. Accounts with tight rate limits
   should lower `OPENAI_GENERATION_OVERPROVISION`.
5. **The auditor is a model, not a proof.** Gates 1–3 and 5 are deterministic
   and cannot regress. Gate 4 is a second model call and is therefore
   probabilistic; it was measured (Section 7) but not proven.
6. **Grounding is not the same as pedagogical quality.** The gates guarantee a
   question is traceable to official content. They do not guarantee it is a
   *good* exam question. Assessing that would need expert review.

---

## 12. Configuration reference

| Variable | Default | Effect |
| --- | --- | --- |
| `OPENAI_API_KEY` | — | Required for all live operations |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `OPENAI_GENERATION_MODEL` | `gpt-5-mini` | Generation and audit model |
| `OPENAI_GENERATION_SHARD_SIZE` | `1` | Questions per concurrent call |
| `OPENAI_GENERATION_OVERPROVISION` | `1.0` | Spare shards, as a fraction of required (floor: 4) |
| `OPENAI_GENERATION_EFFORT` | `low` | Reasoning budget per question |
| `OPENAI_JUDGE_EFFORT` | `minimal` | Reasoning budget for the auditor |
| `OPENAI_GENERATION_CANDIDATES` | `1` | Generations raced inside one shard |
| `READINESS_READY_THRESHOLD` | `80` | Score for *Ready* |
| `READINESS_NEARLY_READY_THRESHOLD` | `60` | Score for *Nearly Ready* |
| `APP_SECRET_KEY` | development value | JWT signing key; refuses to start in production if left at the default |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `1440` | Token lifetime |
| `DATABASE_URL` | `sqlite:///data/assistant.db` | Persistence |
| `CORS_ORIGINS` | localhost:5173 | Allowed browser origins |

---

## 13. Module map

| Path | Responsibility |
| --- | --- |
| `certifications/*.json` | Source-of-truth exam-guide content |
| `src/certification_assistant/ingestion/` | Loading, cleaning, preparation |
| `src/certification_assistant/rag/pipeline.py` | Chunking, embeddings, FAISS, retrieval (the other `rag/` modules re-export from here) |
| `src/certification_assistant/ai/engine.py` | Generation, grounding gates, scheduling |
| `src/certification_assistant/assessment/` | Scoring, weak areas, plans, readiness |
| `src/certification_assistant/schemas.py` | Strict Pydantic contracts |
| `backend/app/api/` | HTTP routes |
| `backend/app/services/` | Orchestration over the AI package |
| `backend/app/models/entities.py` | SQLAlchemy tables |
| `backend/app/core/` | Configuration, database session, auth |
| `frontend/src/components/` | React screens |
| `frontend/src/services/api.js` | API client |
| `tests/test_ai_engine.py` | AI and RAG suite (offline) |
| `backend/tests/test_api.py` | API integration suite |
| `notebooks/01_rag_pipeline.ipynb` | Executed end-to-end evaluation |
| `scripts/run_live_pipeline.py` | Live smoke test against the real API |

---

## 14. Setup

Python 3.11 or newer.

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[notebook,test]"
copy .env.example .env
```

Add a real `OPENAI_API_KEY` to `.env` — never to `.env.example`, which is
committed. `.env` is git-ignored.

Build the retrieval artifacts, then run:

```bash
cert-rag prepare
cert-rag embed
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

```bash
cd frontend && npm install && npm run dev
```

Open `http://127.0.0.1:5173`, register an account, and select AWS Certified AI
Practitioner — the only certification for which generation is enabled.
