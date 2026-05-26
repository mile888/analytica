# Analytica

Analytica is an AI data analysis workspace. It helps a user upload CSV files, ask questions in normal language, see charts, save findings, and build reports.

The agent does not guess numbers. It plans the analysis, then runs real deterministic calculations on the data.

## Quick Start

```bash
git clone <repository-url>
cd analytica-deep-agent
cp .env.example .env
# Add your LLM API key in .env
docker compose up --build
```

Open:

- Frontend: http://localhost:3000
- API docs: http://localhost:8000/docs

## What Analytica Does

Analytica lets you:

- upload one or more CSV datasets;
- ask questions in English or Russian;
- get computed answers, not made-up answers;
- create charts and tables;
- compare datasets safely;
- explain charts;
- save important findings;
- build and export reports.

Example questions:

- "Which cities have the highest total Sales?"
- "Build a histogram of BMI."
- "Which health indicators are most common?"
- "Where is profit margin weakest?"
- "Can these two datasets be joined?"
- "Build a side-by-side visualization for two datasets."

## Main Features

| Feature | What it means |
| --- | --- |
| CSV upload | Upload datasets from the web app. |
| Schema profiling | The backend detects columns, types, and sample values. |
| Natural language analysis | The user asks questions without writing code. |
| Deterministic execution | Pandas and SQL compute the real results. |
| Explicit constraints | If the user says which metric or group to use, the agent keeps it. |
| Business KPIs | The agent can compute profit margin, loss rate, discount sensitivity, and efficiency scores when the fields exist. |
| Health and risk analysis | The agent can rank binary indicators, prevalence, and ordered-group risk. |
| Multi-dataset analysis | Each dataset runs in its own branch, then the answer compares the branch evidence. |
| Charts and tables | Ranking, distribution, trend, relationship, and comparison requests create artifacts. |
| Reports | Findings and artifacts can be turned into a shareable report. |

## How It Works

```text
┌─────────────────────────────────────────────────────────┐
│                    User Question                        │
│           "Top cities by total Sales"                   │
└──────────────────────┬──────────────────────────────────┘
                       ▼
              ┌────────────────┐
              │ Intent Router  │  Understands the task
              └───────┬────────┘
                      ▼
              ┌────────────────┐
              │ Dataset Resolver│  Selects dataset(s)
              └───────┬────────┘
                      ▼
              ┌────────────────┐
              │ Semantic Plan   │  Locks metric, group, filters
              └───────┬────────┘
                      ▼
              ┌────────────────┐
              │ Validator       │  Checks columns and types
              └───────┬────────┘
                      ▼
              ┌────────────────┐
              │ Executor        │  Runs pandas / SQL
              └───────┬────────┘
                      ▼
              ┌────────────────┐
              │ Grounded Answer │  Uses computed evidence only
              └───────┬────────┘
                      ▼
         ┌────────────┴────────────┐
         ▼                         ▼
┌─────────────────┐     ┌──────────────────┐
│ Charts / Tables │     │ Key Findings     │
└────────┬────────┘     └────────┬─────────┘
         └────────────┬──────────┘
                      ▼
              ┌────────────────┐
              │ Report Builder │
              └────────────────┘
```

## Multi-Dataset Flow

```text
User question
  -> multi-dataset planner
  -> one branch per dataset
  -> deterministic branch calculations
  -> evidence package per dataset
  -> grounded comparison
  -> critic checks
```

Each branch keeps its own:

- dataset id;
- metric and group fields;
- derived KPIs;
- computed rows;
- charts and tables;
- limitations.

This avoids mixing unrelated datasets.

## Setup

### Requirements

- Python 3.12+
- Node.js 18+
- Docker and Docker Compose

### Docker

```bash
cp .env.example .env
# Set GEMINI_API_KEY or another configured LLM provider key.
docker compose up --build
```

### Local Backend

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn source.api.app:app --reload
```

### Local Frontend

```bash
cd frontend
npm install
npm run dev
```

## How To Use The App

1. Open the frontend.
2. Upload a CSV dataset.
3. Create an investigation.
4. Ask a question.
5. Review the chart or table.
6. Ask follow-up questions.
7. Save findings.
8. Build a report.
9. Export the report if needed.

## Testing

Run backend tests:

```bash
python3 -m compileall source/
python3 -m pytest -q -m "not integration"
```

Run frontend tests:

```bash
cd frontend
npm test -- --run
npm run build
```

Run Docker build:

```bash
docker compose build
```

Validate the evaluation notebook:

```bash
python3 -m json.tool notebooks/evaluation_metrics.ipynb > /dev/null
```

## Evaluation

The evaluation notebook is here:

```text
notebooks/evaluation_metrics.ipynb
```

It checks:

- end-to-end success;
- analytical correctness;
- chart creation;
- multi-dataset reasoning;
- refusal behavior;
- follow-up behavior.

See also [docs/evaluation.md](docs/evaluation.md).

## Project Structure

```text
analytica-deep-agent/
├── README.md
├── docker-compose.yml
├── Dockerfile.backend
├── requirements.txt
├── pyproject.toml
├── llm_config.yaml
│
├── source/
│   ├── api/                      FastAPI app and routes
│   ├── llm/                      LLM provider setup
│   ├── product/                  Core product logic
│   │   ├── fallback_analysis.py  Deterministic analysis routes
│   │   ├── llm_semantic_planner.py
│   │   ├── business_semantic_planner.py
│   │   ├── dataset_registry.py
│   │   ├── cross_dataset_synthesis.py
│   │   ├── plan_executor.py
│   │   ├── plan_validator.py
│   │   └── sqlite/
│   ├── skills/                   Agent skill prompts
│   └── tools/                    Agent tools
│
├── frontend/
│   ├── app/                      Next.js pages and routes
│   ├── components/               React components
│   └── lib/                      API and UI helpers
│
├── tests/                        Backend tests
├── docs/                         Project documentation
└── notebooks/
    └── evaluation_metrics.ipynb
```

## Architecture Notes

For more detail, read [docs/architecture.md](docs/architecture.md).

Important rules:

- The LLM may plan and explain.
- The LLM must not invent numbers.
- The executor computes all statistics.
- Validators check columns, types, and locked constraints.
- Critics reject ungrounded or incompatible answers.
- Multi-dataset answers must use branch evidence.
- If the data cannot support a request, the agent gives a clear limitation.

## Known Limitations

- CSV files are the main supported input.
- Very large files may need sampling or more memory.
- Correlation is not causation.
- Some unclear questions may need a follow-up clarification.
- Cross-dataset row-level joins only work when safe shared keys exist.
- LLM wording can vary, but computed results should stay stable.

## Mobile And Demo Access

The frontend is responsive. To test on a phone on the same Wi-Fi:

```bash
docker compose up --build
ipconfig getifaddr en0
```

Then open:

```text
http://<your-local-ip>:3000
```

## Future Work

- More chart types.
- Larger dataset support.
- Direct database connections.
- Shared team workspaces.
- More report templates.

## Author

Analytica was built as a research project about AI-assisted data investigation.
