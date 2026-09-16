# Development guide

This document contains the setup, validation, and repository-reference details
that are intentionally kept out of the project showcase in `README.md`.

## Requirements

- Python 3.11
- Node.js 22.12+ (22.x)
- Docker Compose v2 (optional, recommended for the complete local stack)
- Poppler and Tesseract when parsing scanned PDFs or images outside Docker

## Docker Compose

From the repository root:

```bash
cp .env.example .env
# Configure the model providers and set a non-empty POSTGRES_PASSWORD.
docker compose up --build -d --wait
```

Open:

- Web: <http://localhost:4001>
- API documentation: <http://localhost:8001/docs>

The bundled topology binds the Web and API to loopback and does not publish the
PostgreSQL port. It deliberately runs one backend worker and one replica because
the embedded Chroma data plane is not multi-process safe. See
[`ARCHITECTURE.md`](ARCHITECTURE.md) for the execution and deployment boundaries
and [`CONFIGURATION.md`](CONFIGURATION.md) for provider and readiness settings.

Stop the stack with:

```bash
docker compose down
```

The one-shot initialization container also repairs ownership on Chroma volumes
created by earlier root-based images. A normal upgrade therefore does not
require deleting the existing local index volume.

## Local development

Create the backend environment and start FastAPI:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env
# Configure the model providers in .env.
python -m uvicorn main:app --reload --port 8001 --workers 1 --no-access-log
```

Start the frontend in a second terminal:

```bash
cd frontend
npm ci
npm run dev
```

The development frontend runs at <http://localhost:5173> and proxies API
requests to the backend.

For optional local Cross-Encoder reranking, install the larger runtime and set
`RERANKER_ENABLED=true`:

```bash
pip install -r requirements-reranker.txt -r requirements-dev.txt
```

The default install and Docker image intentionally omit this large model
runtime.

## Validation

Backend static checks and deterministic tests:

```bash
python -m ruff check .
python -m pytest -q --cov --cov-config=pyproject.toml --cov-report=term-missing:skip-covered
```

PostgreSQL-specific state and recovery tests require a disposable database that
the test suite is allowed to clear:

```bash
TEST_DATABASE_URL=postgresql://user:password@127.0.0.1:5432/studyloop_test \
  python -m pytest -q \
    tests/test_idempotency_postgres.py \
    tests/test_quiz_sessions_postgres.py \
    tests/test_adaptive_sessions_postgres.py \
    tests/test_autonomous_sessions_postgres.py \
    tests/test_memory_postgres.py
```

Deterministic tool-loop demonstration:

```bash
python scripts/demo_react_tutor_agent.py
```

Frontend validation:

```bash
cd frontend
npm run lint
npm run build
npm run test:e2e
```

Install Playwright's Chromium runtime once before the first local E2E run:

```bash
npx playwright install chromium
```

Browser E2E uses the local Vite application and controlled API fixtures; it
does not call a model provider. CI also validates Docker Compose security
defaults, builds both images, and runs a storage/readiness smoke test. The core
Python runtime branch-coverage regression floor is configured in
`pyproject.toml` and enforced by CI.

## Project structure

```text
study-loop/
├── agents/          LangGraph agents and workflows
├── routers/         FastAPI routes
├── services/        retrieval, model, memory, and tool services
├── models/          Pydantic data models
├── frontend/        React application and Playwright E2E
├── tests/           backend tests
├── examples/        example learning material
└── docs/            architecture and operating documentation
```

