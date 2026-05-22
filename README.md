# Analytica Deep Agent

Analytica is an execution-first analytical investigation workspace for arbitrary tabular datasets. It lets a user upload data, ask analytical questions, get real computed results, review artifacts, continue analysis across branches, and turn evidence-backed findings into reports.

The project is intentionally dataset-agnostic. Runtime logic must resolve metrics, dimensions, timestamps, filters, transformations, and relationships from the active schema/profile and executable rows. Dataset-specific column names and sample values belong only in tests or clearly marked examples.

## Why It Is Useful

Analytical assistants often understand intent but drift into generic narration when execution context is missing or a target cannot be resolved. Analytica is built to be honest about that boundary: executable questions either run against real rows and materialize artifacts, or return a clear limitation.

Core strengths:

- schema-driven metric, dimension, timestamp, and value resolution;
- execution-context checks before dataframe analysis;
- artifact-first answers for charts, tables, transformations, and reports;
- branch-aware follow-ups that preserve analytical lineage;
- no fake chart generation from profile-only metadata;
- reviewable final reports and published report snapshots.

## Main User Flow

1. Upload or attach a CSV dataset through the API or frontend.
2. Analytica profiles schema, stores execution metadata, and marks whether row-level execution is available.
3. The user asks a question such as a ranking, aggregation, histogram, trend, comparison, transformation, or relationship test.
4. The planner resolves intent and schema targets.
5. Runtime resolution verifies that raw rows are available and loadable.
6. The executor computes real results and creates grounded artifacts.
7. Follow-up questions continue from the active branch or artifact when appropriate.
8. Findings can be reviewed, promoted into a report, finalized, and published.

## Architecture Overview

- `source/agent.py` builds the official DeepAgents SDK runtime.
- `source/tools/analytics_tools.py` exposes controlled schema, Python, SQL, visualization, and report tools.
- `source/product/execution_context.py` records dataset runtime availability and reconstructs executable dataframes from persistent storage.
- `source/product/execution_planner.py` produces structured, schema-driven query plans.
- `source/product/direct_query_executor.py` executes deterministic pandas analyses for resolved plans.
- `source/product/branch_workspace.py` tracks branch identity and chart/report lineage.
- `source/product/fallback_analysis.py` repairs failed/empty runner output without pretending execution happened.
- `source/product/sqlite/` stores investigations, data sources, reports, artifacts, and runtime metadata.
- `source/api/` exposes the FastAPI product backend.
- `frontend/` provides the Next.js investigation workspace.
- `pages/` and `app.py` provide a Streamlit demo client.

See [docs/architecture.md](docs/architecture.md) for a fuller map.

## Deep Agent Usage

Analytica uses `create_deep_agent` from the DeepAgents SDK with:

- typed runtime context via `AnalyticaContext`;
- project skills from `source/skills/*/SKILL.md`;
- analytical tools from `source/tools/analytics_tools.py`;
- filesystem-backed artifacts and memory;
- optional LangGraph checkpoint persistence.

DeepAgents handles model/tool orchestration, skill selection, virtual filesystem access, and conversation continuity. Analytica-specific code stays focused on deterministic data execution, artifact validation, product persistence, and report workflows.

Analytica intentionally does not define custom subagents. Branches, reports, findings, and artifacts are product-layer state, not separate agents. This keeps execution-context safety, artifact grounding, and no-fake-analytics checks deterministic and testable.

## Planner, Executor, Branches, And Artifacts

The planner resolves analytical intent into a structured plan: metric, dimension, filters, time axis, aggregation, transformation, chart type, and confirmation requirements. It does not silently substitute a requested target when the schema does not support it.

Before execution, dataset runtime resolution verifies that raw rows exist and can be loaded. If execution is unavailable, Analytica returns an execution-context error and does not create placeholder artifacts.

The direct executor computes grounded results with pandas. Artifacts are created only from computed rows, including validated histogram bins and comparison payloads.

Branch state records active metric, dimension, aggregation, filters, chart type, parent artifact, and derived artifact metadata so follow-ups such as transformed-chart explanations bind to the correct lineage.

## Setup

```bash
make setup
make update
cp .env.example .env
```

Configure an LLM provider in `.env` using the environment variable named in `llm_config.yaml`, for example:

```bash
GEMINI_API_KEY="..."
ANALYTICA_LLM_PROVIDER="gemini"
```

Useful local settings:

```bash
ANALYTICA_DEFAULT_DATA_PATH="data"
ANALYTICA_ARTIFACT_DIR="artifacts"
ANALYTICA_SQL_TABLE_NAME="data"
ANALYTICA_CHECKPOINTER_TYPE="memory"
```

`ANALYTICA_DEFAULT_DATA_PATH` is only a local CLI/Streamlit convenience. The primary product flow is dataset upload through the API or frontend.

## Run Backend

FastAPI:

```bash
uvicorn source.api.app:app --reload
```

Streamlit demo:

```bash
streamlit run app.py
```

CLI:

```bash
python3 run.py --csv path/to/file.csv --query "Summarize this dataset and suggest analysis directions"
```

## Run Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the local Next.js URL shown by the dev server. The frontend expects the FastAPI backend to be running.

## Docker Local Run

Analytica can run with a backend container, a frontend container, and named Docker volumes for local runtime data.

```bash
cp .env.example .env
docker compose up --build
```

Open:

- frontend: http://localhost:3000
- backend API/docs: http://localhost:8000/docs
- backend health check: http://localhost:8000/health

Put provider keys such as `GEMINI_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, or `OPEN_API_TOKEN` in `.env` locally, or in the deployment provider's environment variable dashboard. Do not commit real keys.

The compose setup mounts persistent volumes for:

- backend runtime state: `/app/.analytica`
- generated artifacts: `/app/artifacts`

Stop containers with:

```bash
docker compose down
```

Clear local Docker runtime state only when you intentionally want a fresh workspace:

```bash
docker compose down -v
```

Server-side frontend requests use `INTERNAL_API_URL`. In Docker Compose this should be `http://backend:8000` because the frontend container reaches the backend by service name. Browser-facing requests use `NEXT_PUBLIC_API_URL`; for local Docker it should stay `http://localhost:8000`. The backend CORS allowlist is controlled by `ANALYTICA_CORS_ORIGINS`.

## Deploy To Render Or Railway

For Render:

1. Create a backend web service from this repository using `Dockerfile.backend`.
2. Set the backend start port with `PORT`; the Dockerfile starts `uvicorn source.api.app:app` on `0.0.0.0`.
3. Add provider keys and Analytica variables in the Render dashboard.
4. Configure a persistent disk mounted at `/app/.analytica` if uploaded datasets and runtime state must survive restarts.
5. Create a frontend web service using `frontend/Dockerfile`.
6. Set `NEXT_PUBLIC_API_URL` to the public backend URL, set `INTERNAL_API_URL` to the backend service URL when the platform provides one, and set `ANALYTICA_CORS_ORIGINS` on the backend to include the public frontend URL.

For Railway:

1. Create a project from the GitHub repository.
2. Add separate backend and frontend services, pointing them at `Dockerfile.backend` and `frontend/Dockerfile`.
3. Set service variables for model provider keys, `INTERNAL_API_URL`, `NEXT_PUBLIC_API_URL`, and `ANALYTICA_CORS_ORIGINS`.
4. Generate public domains for both services.
5. Use a Railway volume for `/app/.analytica` if runtime uploads should persist.

No Kubernetes, reverse proxy, or custom domain is required for the simple coursework deployment.

## Tests And Checks

Backend:

```bash
python3 -m compileall .
python3 -m pytest -q
```

Frontend:

```bash
cd frontend
npm run typecheck
npm test -- --run
npm run build
```

Notebook files are intentionally excluded from cleanup and should not be modified during repository maintenance.

## Demo Script

1. Start the backend with `uvicorn source.api.app:app --reload`.
2. Start the frontend with `cd frontend && npm run dev`.
3. Upload a CSV dataset.
4. Ask for a grouped ranking, a histogram, and a temporal trend.
5. Ask a transformation follow-up such as removing outliers.
6. Explain the adjusted chart and inspect the generated artifacts.
7. Promote findings into a report and finalize a published snapshot.

## Known Limitations

- External warehouse/database connections are not exposed; SQL is scoped to the active dataframe.
- Relationship analysis computes requested numeric pairs and simple time-based growth; causal interpretation remains out of scope.
- Semantic aliases are generic and schema-validated, but ambiguous business vocabulary may still require user confirmation.
- Streamlit is retained as a demo client; the primary product surface is FastAPI plus Next.js.
