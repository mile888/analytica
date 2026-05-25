# Analytica Architecture

Analytica is organized around an API-first analytics product model. The FastAPI backend owns persistence, execution safety, dataset scope, artifacts, and reports. The Next.js workspace is the only product UI.

Runtime logic is dataset-agnostic. Metrics, dimensions, timestamps, identifiers, chart intent, and data quality concerns are inferred from uploaded dataset profiles, attached dataset registries, artifact lineage, and executable dataframe runtimes.

## System Overview

```
┌─────────────────────────────────────┐
│           Next.js Frontend          │
│  ┌─────────┐ ┌──────┐ ┌──────────┐ │
│  │ Chat    │ │Visual│ │ Report   │ │
│  │Workspace│ │Panel │ │ Editor   │ │
│  └────┬────┘ └──┬───┘ └────┬─────┘ │
│       └─────────┼──────────┘       │
└─────────────────┼──────────────────┘
                  │ REST API
┌─────────────────┼──────────────────┐
│          FastAPI Backend           │
│  ┌──────────────┴───────────────┐  │
│  │      API Routes Layer        │  │
│  │  /investigations  /reports   │  │
│  │  /data-sources    /runs      │  │
│  └──────────────┬───────────────┘  │
│  ┌──────────────┴───────────────┐  │
│  │     Product Services Layer   │  │
│  │  InvestigationRunService     │  │
│  │  InvestigationService        │  │
│  │  ReportEditingService        │  │
│  └──────┬───────────┬───────────┘  │
│  ┌──────┴─────┐ ┌───┴──────────┐   │
│  │Deterministic│ │ DeepAgents  │   │
│  │ Executor   │ │ LLM Runtime │   │
│  └──────┬─────┘ └───┬─────────┘   │
│  ┌──────┴───────────┴──────────┐   │
│  │    SQLite Persistence       │   │
│  │  Investigations │ Artifacts │   │
│  │  Reports        │ Runtime   │   │
│  └─────────────────────────────┘   │
└────────────────────────────────────┘
```

## Investigation Pipeline

Every user question flows through a structured pipeline:

```
User Question
     │
     ▼
┌────────────────────┐
│  Intent Resolution  │  Classify: ranking, histogram, trend,
│                     │  hypothesis, comparison, transformation,
│                     │  report, cross-dataset, follow-up
└─────────┬──────────┘
          ▼
┌────────────────────┐
│  Dataset Resolution │  Score candidates by:
│                     │  - metric compatibility
│                     │  - schema match
│                     │  - entity alignment
│                     │  - branch/artifact lineage
│                     │  - recent context
└─────────┬──────────┘
          ▼
┌────────────────────┐
│ Execution Planning  │  Resolve:
│                     │  - metric column
│                     │  - dimension column
│                     │  - filter values
│                     │  - aggregation type
│                     │  - chart intent
└─────────┬──────────┘
          ▼
┌────────────────────┐
│    Execution        │  Deterministic pandas
│                     │  computation on real data
└─────────┬──────────┘
          ▼
┌────────────────────┐
│   Validation        │  Quality gate checks:
│                     │  - correct language
│                     │  - no hallucinated data
│                     │  - artifact grounding
│                     │  - transformation respect
└─────────┬──────────┘
          ▼
   ┌──────┴──────┐
   ▼              ▼
┌────────┐  ┌──────────┐
│Artifacts│  │ Findings │
│ Charts  │  │ Insights │
│ Tables  │  │ Scores   │
└────┬───┘  └────┬─────┘
     └─────┬─────┘
           ▼
    ┌──────────────┐
    │    Report     │
    │  (optional)   │
    └──────────────┘
```

## Multi-Dataset Reasoning

When multiple datasets are attached, the system performs cross-dataset analysis:

```
Attached Datasets
    │
    ▼
┌──────────────────────┐
│ Semantic Profiling    │  For each dataset:
│                      │  - column roles (metric/dimension/timestamp/id)
│                      │  - domain purpose (sales/HR/marketing/ops)
│                      │  - entity types (customer/product/location)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Relationship Analysis │  Between datasets:
│                      │  - shared entity detection
│                      │  - joinability scoring
│                      │  - overlap measurement
│                      │  - complementarity assessment
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Scope Resolution     │  For the user's query:
│                      │  - single-dataset → direct execution
│                      │  - cross-dataset  → comparative analysis
│                      │  - ambiguous      → clarification request
└──────────────────────┘
```

## Folder Map

```
source/
  agent.py                 DeepAgents runtime construction and invocation
  runtime_context.py       Typed runtime context for DeepAgents tools
  tools/                   LangChain/DeepAgents-compatible analytics tools
  skills/                  DeepAgents skill folders

  api/
    app.py                 FastAPI app assembly
    deps.py                Shared API dependencies
    serialization.py       Product model serialization helpers
    routes/                Thin HTTP routes for product objects

  product/
    data_sources.py        Data source, profile, and usage context models
    dataset_registry.py    Investigation dataset registry and resolver
    data_profiling.py      Lightweight pandas profiling
    data_context.py        Compact context builder for agent runs
    semantic_layer.py      Shared semantic profile and focus helpers
    evaluation_api.py      Public evaluation API for benchmark notebooks
    investigation.py       Investigation, message, run, artifact, report models
    run_service.py         API-driven run orchestration
    direct_query_executor.py  Deterministic dataframe execution
    branch_workspace.py    Branch identity and chart/report lineage
    report_builder.py      Structured investigation report builder
    report_artifacts.py    Stable report artifact snapshots
    report_service.py      Editing, review, approval, final snapshot service
    exporter.py            TXT, Markdown, HTML, and PDF report export
    sqlite/                SQLite persistence and serialization

frontend/                  Next.js investigation workspace
tests/                     Backend and product contract tests
notebooks/                 Evaluation notebooks
docs/                      Product and architecture documentation
```

## Backend Layers

```
FastAPI routes
  → product services
  → dataset/runtime resolution
  → DeepAgents runtime or deterministic dataframe executor
  → product store
  → SQLite or in-memory persistence
```

Route files stay thin. Business behavior belongs in `source/product/`, and agent-specific execution belongs in `source/agent.py` plus DeepAgents tools and skills.

## Dataset Scope and Execution Safety

Every analytical plan resolves dataset scope before execution. Scope can be a single dataset, multiple datasets, a cross-dataset comparison, or an ambiguity that requires clarification. The resolver scores explicit dataset mentions, schema matches, value matches, semantic roles, branch/artifact lineage, recent context, user selection, and runtime availability.

Execution is blocked when dataset runtime is unavailable or requested fields cannot be resolved. The system does not fall back to the first dataset, hallucinate columns, or create placeholder artifacts.

## Agent Framework

`source/agent.py` uses the DeepAgents SDK:

```python
from deepagents import create_deep_agent
```

The agent is configured with LangChain `@tool` analytics tools, explicit skill folders, filesystem-backed artifacts and memory, typed runtime context, and optional checkpointing.

`AuthoritativeExecutionPlanner` returns query plans, aliases, filters, chart intent, and validation metadata. It does not synthesize final analytical prose. `BranchWorkspaceManager` consumes plans and stores continuation metadata without independently reinterpreting raw user text.

## Run Lifecycle

```
queued
  → preparing_data
  → building_context
  → running_analysis
  → validating_results
  → generating_report
  → completed / failed
```

Run events are user-facing timeline entries and support cursor-based polling.

## Report Pipeline

```
Investigation workspace
  → ShareableReport draft
  → human edits and section review
  → comments and approval
  → readiness checks
  → immutable FinalReportSnapshot
  → PDF export with embedded chart images
```

## SQL Scope

SQL is scoped to executable dataframe runtimes. The active dataframe is copied into an in-memory SQLite table and queried through checked read-only SQL. External warehouse connections are future work.

## Persistence

SQLite persistence lives under `source/product/sqlite/`. Schema changes are managed by migrations in `source/product/migrations.py`. Runtime datasets and uploaded files are local deployment state, not source-controlled artifacts.

## Frontend

The product UI is the Next.js workspace in `frontend/`. It uses `frontend/lib/api.ts` for backend calls and keeps client state focused on navigation, active branches, visual artifacts, report preview, and export actions.
