# Analytica Architecture

Analytica is organized around an API-first analytics product model. Streamlit remains a temporary demo client, while the FastAPI backend and the Next.js workspace provide the main product surface for persistent analytical investigations.

Analytica is dataset-agnostic. Runtime logic should infer metrics, dimensions,
timestamps, identifiers, chart intent, and data quality concerns from the
uploaded dataset profile or raw dataframe. Product code must not depend on
specific sample columns or filenames; sample names belong only in tests or
clearly labeled examples.

## Folder Map

```text
source/
  agent.py                 DeepAgents runtime construction and invocation helpers
  runtime_context.py       Typed runtime context passed to DeepAgents tools
  tools/                   Small LangChain/DeepAgents-compatible analytics tools
  skills/                  Official DeepAgents skill folders, each with SKILL.md

  api/
    app.py                 FastAPI app assembly
    deps.py                Shared API dependencies
    serialization.py       Product model serialization helpers
    routes/                Thin HTTP routes for product objects

  product/
    data_sources.py        Data source, profile, semantic note, and usage context models
    data_profiling.py      Lightweight pandas profiling
    data_context.py        Compact context builder for agent runs and UI previews
    semantic_layer.py      Shared semantic dataset profile and investigation focus helpers
    question_suggestions.py Dataset-agnostic prompt suggestion helpers
    investigation.py       Investigation, message, run, artifact, report, review, and final snapshot models
    service.py             Investigation service compatibility layer
    run_service.py         API-driven run orchestration
    run_validation.py      Lightweight run result validation
    event_stream.py        Cursor helpers for polling-friendly run events
    report_builder.py      ShareableReport builders
    report_service.py      Editing, review, approval, and final snapshot service logic
    readiness.py           Report readiness checks
    final_report_registry.py Final snapshot listing/search helpers
    exporter.py            Markdown/HTML export
    store.py               In-memory store and public store contract
    sqlite_store.py        Compatibility wrapper for SQLite store imports
    sqlite/                SQLite implementation and serialization helpers
    store_factory.py       Environment-driven store selection
    migrations.py          SQLite schema migrations
    file_storage.py        Local uploaded CSV storage

pages/                     Streamlit demo/client entrypoints
source/ui/streamlit/       Streamlit page implementation modules
frontend/                  Next.js Decision Workspace shell
tests/                     Backend and product contract tests
docs/                      Product and architecture documentation
```

## Backend Layers

The product backend follows this path:

```text
FastAPI routes
  -> product services
  -> DeepAgents runtime / pandas helpers
  -> product store
  -> SQLite or in-memory persistence
```

Route files should stay thin. Business behavior belongs in `source/product`, and agent-specific execution belongs in `source/agent.py` plus official DeepAgents tools and skills.

## Semantic Analytical Layer

Analytica now has a shared semantic layer in `source/product/semantic_layer.py`.
It normalizes dataframe/profile/context metadata into `SemanticDatasetProfile`
and `SemanticColumnProfile` objects. The same inferred roles power chart
selection, deterministic fallback analysis, and data-aware question suggestions.

The layer infers metrics, dimensions, timestamps, identifiers, and text fields
from dtype, cardinality, examples, semantic notes, and dataset-agnostic naming
signals. It also tracks lightweight `InvestigationFocus` from the latest chart
or analysis context so ambiguous follow-ups such as "which are strongest?" can
continue the active metric/dimension instead of drifting to an unrelated column.

Explicit user wording always wins over stale focus. Sample dataset names and
specific columns must remain test fixtures or examples, not product logic.

## Agent Framework Alignment

`source/agent.py` uses the official DeepAgents SDK harness:

```python
from deepagents import create_deep_agent
```

The agent is configured through documented constructor parameters rather than a
custom LangGraph node graph:

- `tools=[...]` receives LangChain `@tool` analytics tools.
- `skills=[...]` points DeepAgents at explicit skill source folders such as
  `/source/skills/data-analysis/`, `/source/skills/csv-dataframe-analysis/`,
  `/source/skills/visualization/`, `/source/skills/business-analysis/`, and
  `/source/skills/reporting/`.
- `memory=["/memories/AGENTS.md"]` exposes filesystem-backed memory.
- `backend=CompositeBackend(...)` routes artifacts, memory, and skills through
  DeepAgents filesystem backends.
- `context_schema=AnalyticaContext` passes runtime/session values through typed
  `ToolRuntime[AnalyticaContext]` instead of globals.

DeepAgents already provides the harness loop, planning support, filesystem
tools, backend routing, memory, context propagation, and progressive skill
loading. Analytica keeps custom code focused on product-specific analytics
tools, deterministic validation, data-source persistence, and report workflows.
It passes pre-constructed backend instances rather than backend factories.

Analytica does not add custom subagents. The official Deep Agents default
subagent support is left to the SDK, while application concepts such as
branches, findings, artifacts, and reports remain deterministic product-layer
state. That separation prevents agent delegation from bypassing execution
context checks, artifact validation, or no-fake-fallback enforcement.

`AuthoritativeExecutionPlanner` is a structured planning helper exposed through
DeepAgents tools. It returns query plans, aliases, filters, chart intent, and
validation metadata; it does not synthesize final analytical prose. The
`BranchWorkspaceManager` consumes those plans and stores branch metadata only.
It decides create/switch/continue from plan identity and does not reinterpret
raw user text.

The deterministic fallback remains as a safety path for runner failures, empty
answers, and artifact repair in the product service. It is trace-marked as a
fallback and should not be treated as the primary orchestration layer; normal
analytical execution is routed through the DeepAgents runner.

## SQL Scope

The current SQL capability is DataFrame SQL: the active dataframe is copied into
an in-memory SQLite table and queried through read-only checked SQL. This follows
the safety spirit of LangGraph SQL-agent guidance by limiting permissions and
checking queries before execution, but it is not an external database agent.

External SQL/warehouse connections are future work. Until permission scoping,
connection management, query review, and product UX are implemented, docs and UI
should describe SQL as advanced DataFrame querying only.

## Persistence

SQLite persistence lives under `source/product/sqlite/`. The public compatibility import remains:

```python
from source.product.sqlite_store import SQLiteInvestigationStore
```

This keeps old tests, Streamlit pages, and external imports stable while allowing the SQLite implementation to be split internally.

Schema changes are managed by migrations in `source/product/migrations.py`. The current schema version should only change when the database schema changes.

## Run Lifecycle

Investigations are executed through `InvestigationRunService`:

```text
queued
  -> preparing_data
  -> building_context
  -> running_analysis
  -> validating_results
  -> generating_report
  -> completed / failed
```

Run events are user-facing timeline entries. They are intentionally separate from raw agent diagnostic logs and support polling through cursor-based event responses.

Follow-up questions are stored as `InvestigationMessage` records. A run can
reference a `message_id`, making that message the active question while keeping
recent messages, previous findings, artifact titles, and the latest report
summary as compact context.

## Report Lifecycle

Reports move through this flow:

```text
Investigation workspace
  -> ShareableReport draft
  -> human edits and section review
  -> comments and approval
  -> readiness checks
  -> immutable FinalReportSnapshot
  -> Reports page / export
```

Draft exports render the current report. Final report snapshots store immutable Markdown/HTML content and approval/readiness metadata at the moment of publication.

## Clients

Streamlit is still useful for demos and local product inspection, but it should not own core product logic. Its large workspace page is implemented under `source/ui/streamlit/investigation_workspace/` and exposed through the thin Streamlit entrypoint in `pages/Investigation_Workspace.py`.

Next.js is a thin Decision Workspace shell. It uses `frontend/lib/api.ts` for API calls and avoids heavy client-side state management. The shell validates that the backend product model works outside Streamlit.
