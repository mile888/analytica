# Investigation Workspace

For the repository-level folder and layer map, see
[architecture.md](architecture.md).

Analytica is gaining a product layer centered on an `Investigation`, not a chat
session. An Investigation is a user-owned analytical workspace for moving from a
business question to reviewed artifacts and a DecisionReport.

## Product model

The core lifecycle is:

```text
question -> investigation -> artifacts -> report -> review
```

An Investigation represents an analytical question such as:

- "Summarize this dataset and identify useful analysis directions."
- "Which groups or categories differ the most?"
- "Are there unusual values, anomalies, or data quality issues?"

Each Investigation keeps:

- `title`
- `user_question`
- `status`
- `created_at` / `updated_at`
- `data_sources`
- `runs`
- `artifacts`
- `findings`
- `report`
- `trace`

## Product objects

- `Investigation`: the durable workspace object for one analytical task.
- `InvestigationRun`: one execution attempt for an Investigation.
- `Artifact`: a generated or captured object such as a table, chart, SQL,
  Python code, text result, report, or validation trace.
- `Finding`: a product-facing insight or limitation that should be reviewed.
- `DecisionReport`: the user-facing report that summarizes evidence, findings,
  limitations, and next steps.
- `ShareableReport`: a clean final document generated from an Investigation for
  external sharing.
- `RunTrace`: execution history stored as trace events and validation artifacts.

## Relationship to the current agent

The existing DeepAgents runtime remains the analytical engine. The product
layer wraps the current runner without changing safe execution, artifacts,
checkpointing, or the existing Streamlit demo.

The agent is created with the official DeepAgents SDK entrypoint:

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    model=...,
    tools=...,
    system_prompt=...,
    skills=[
        "/source/skills/data-analysis/",
        "/source/skills/csv-dataframe-analysis/",
        "/source/skills/visualization/",
        "/source/skills/business-analysis/",
        "/source/skills/reporting/",
    ],
    memory=["/memories/AGENTS.md"],
    backend=CompositeBackend(
        default=StateBackend(),
        routes={
            "/artifacts/": FilesystemBackend(root_dir="artifacts", virtual_mode=True),
            "/memories/": FilesystemBackend(root_dir=".analytica/memory", virtual_mode=True),
            "/source/skills/": FilesystemBackend(root_dir="source/skills", virtual_mode=True),
        },
    ),
    checkpointer=...,
)
```

Skills are regular DeepAgents skill folders under `source/skills/*/SKILL.md`.
There is no custom `load_skill` tool or manual `SKILL.md` registry in the agent
runtime; DeepAgents injects its Skills System and reads relevant skills through
the configured backend.

## DeepAgents context and memory

The agent now follows the official DeepAgents context engineering pattern:

- thread-scoped scratch files stay in the default `StateBackend`
- durable agent memory is exposed at `/memories/`
- physical memory files live under `.analytica/memory/`
- long artifacts can be exposed through `/artifacts/`
- skills are exposed through the filesystem backend at `/source/skills/` and
  passed as explicit source directories containing `SKILL.md`
- large intermediate outputs should be written to filesystem artifacts instead
  of being repeatedly inserted into the prompt

The default memory file is `/memories/AGENTS.md`, backed by
`.analytica/memory/AGENTS.md`. It is initialized with a short scaffold and is
intended only for durable, high-signal facts such as reusable analysis guidance,
stable user preferences, or known dataset caveats. Raw message history,
temporary tool output, and large reports should not be stored there.

Context management is left to DeepAgents and LangGraph checkpointing. The app no
longer performs custom conversation-history trimming before calling the agent.

Runtime-specific data is passed through the official typed DeepAgents runtime
context pattern. `source.runtime_context.AnalyticaContext` is registered via
`context_schema=AnalyticaContext`, and `run_agent` / `run_agent_stream` pass an
instance through `agent.invoke(..., context=...)` or
`agent.stream(..., context=...)`. Analytics tools that need run-specific state
receive `runtime: ToolRuntime[AnalyticaContext]` and read values from
`runtime.context`, rather than from the system prompt.

Custom analytics tools follow the official LangChain / DeepAgents tool style:
they are `@tool`-decorated `BaseTool` objects passed through
`create_deep_agent(..., tools=[...])`. Their public argument schemas stay small;
`runtime` is injected by LangGraph and does not appear as a model-facing tool
argument. The agent gets built-in DeepAgents filesystem, todo, summarization,
skills, memory, and subagent behavior from the SDK instead of duplicate custom
wrappers.

## Planner, Branches, And Fallbacks

Analytica is a Deep Agent-native analytical copilot:

- Deep Agent orchestrates work, chooses tools, uses skills, manages
  filesystem/context, and synthesizes final answers.
- Tools compute deterministic results, create charts/artifacts, and validate
  correctness.
- `AuthoritativeExecutionPlanner` builds structured query plans, preserves
  filters and aliases, keeps chart intent, and blocks unsafe substitutions.
- `BranchWorkspaceManager` stores branch metadata for multiple analytical
  branches inside one investigation. It consumes query plans and does not parse
  raw user text independently.
- The product layer handles API/session/persistence/display. Its deterministic
  fallback is a safety mechanism for runner errors, empty answers, or missing
  artifacts, not a second reasoning engine.

`InvestigationService` runs the existing agent through dependency injection. The
default runner calls the current `source.agent.run_once` function. Agent output
is translated by `agent_output_to_investigation_update`, which maps summaries,
findings, generated code, SQL metadata, artifacts, limitations, and tool
timeline into product objects.

## Current implementation

- `source/api/`: FastAPI product backend and thin route handlers.
- `source/product/investigation.py`: product domain models.
- `source/product/data_sources.py`: Data Source Registry models.
- `source/product/data_profiling.py`: lightweight pandas profiling.
- `source/product/data_context.py`: compact usage context for agent/UI.
- `source/product/semantic_layer.py`: shared semantic column understanding and
  active investigation focus.
- `source/product/question_suggestions.py`: universal and data-aware question suggestions.
- `source/product/run_service.py`: API-driven InvestigationRun orchestration.
- `source/product/event_stream.py`: polling-friendly run event cursors.
- `source/product/report_builder.py`, `report_service.py`, `readiness.py`: report, review, versioning, and readiness workflows.
- `source/product/final_report_registry.py`: final snapshot listing helpers.
- `source/product/store.py`: in-memory `InvestigationStore`.
- `source/product/sqlite_store.py`: persistent SQLite store.
- `source/product/store_factory.py`: store backend selection from environment.
- `source/product/migrations.py`: SQLite schema migrations.
- `source/product/exporter.py`: TXT, Markdown, and HTML export helpers.
- `pages/`: Streamlit temporary/demo clients.
- `frontend/`: thin Next.js Decision Workspace shell over the FastAPI backend.
- `source/product/data_sources.py`: Data Source Registry domain models.
- `source/product/data_profiling.py`: lightweight pandas profiling for data
  sources.
- `source/product/report_builder.py`: ShareableReport generation from
  Investigations.
- `source/product/report_service.py`: editable ShareableReport workflow and
  version snapshots.
- `source/product/readiness.py`: lightweight report readiness checks.
- `source/product/final_report_registry.py`: final snapshot listing and
  summaries.
- `source/product/adapter.py`: robust adapter from agent output to product
  artifacts, findings, report, and trace.
- `source/product/service.py`: service layer connecting the store to the agent
  runner.
- `source/api/*`: minimal product API foundation for investigations and
  reports.
- `pages/Investigation_Workspace.py`: temporary Streamlit workspace UI.
- `pages/Data_Sources.py`: temporary Streamlit Data Source Registry UI.

The in-memory store remains available for tests and fallback. The app defaults
to SQLite persistence.

## Semantic analytical continuity

The workspace uses one shared semantic understanding layer for fallback
analysis, chart intent, and suggested questions. It builds a
`SemanticDatasetProfile` from the raw dataframe or saved data-source profile,
then infers which columns behave like metrics, grouping dimensions, time axes,
identifiers, or text fields.

Follow-up analysis also receives a compact `InvestigationFocus` derived from the
latest chart and recent analytical outputs. This lets ambiguous questions keep
working against the current analytical thread, while explicit wording in the new
question still overrides stale context.

Structured findings carry report-ready metadata such as conclusion, confidence,
evidence strength, limitation, business implication, and recommended validation.
Reports prefer those structured fields over raw metadata or workflow details.

## API-first architecture

The product is moving toward:

```text
FastAPI product backend
    -> product services
    -> agent runtime
    -> SQLite product store
```

Streamlit remains a temporary client shell. Product objects, persistence,
readiness checks, report publishing, and data source management live in
`source/product/*` and are exposed through FastAPI routes. This keeps the future
frontend migration path clean without rewriting the agent runtime.

The API is still intentionally simple: no auth, no async job system, no
websocket streaming, and no enterprise governance. The goal is to make FastAPI
the primary product backend one slice at a time.

## Data Source Registry

Data sources are now first-class product objects. A `DataSource` represents a
CSV, SQLite, Postgres, DuckDB, or unknown source that can be linked to
Investigations.

Data source fields include:

- `id`
- `name`
- `type`
- `created_at`
- `updated_at`
- `status`
- `location`
- `description`
- `tags`
- `linked_investigation_ids`
- `metadata`

Data source statuses are:

- `active`
- `stale`
- `error`
- `archived`

Each source can have a `DataSourceProfile` with row/column counts, column
metadata, missing-value summary, numeric stats, categorical top values, sampled
rows, and a generated timestamp. Profiling uses pandas only and is designed to
be lightweight and robust rather than exhaustive.

Investigations now carry `linked_data_source_ids`. This creates a durable link
between an analytical question and the data sources used to answer it.

The temporary Streamlit page lives at:

```text
pages/Data_Sources.py
```

It supports CSV upload, source creation, profile display, archiving, and linking
sources to Investigations. This is productized data management, not just a raw
file upload widget.

## Data Source Usage Context

`DataSourceProfile` is raw descriptive metadata: shape, columns, missingness,
numeric summaries, categorical summaries, and sample rows.

`DataSourceUsageContext` is the compact product context used by the agent and
UI. It is derived from the source, profile, linked Investigations, and source
status. It includes:

- schema summary
- inferred column roles
- sampled rows
- missing, numeric, and categorical summaries
- freshness/status information
- linked Investigation ids
- previous questions asked against the source
- known caveats

Column roles are inferred with simple deterministic rules, not LLM semantics:
datetime-like names/dtypes become `timestamp`; numeric columns become `metric`;
low-cardinality strings become `dimension`; id/name/code-like high-cardinality
columns become `identifier`; long free-form strings become `text`; uncertain
columns remain `unknown`.

This helps the agent reason about the dataset before writing analysis code and
helps users see what the product believes about a source. It prepares the
system for a future Decision Workspace frontend without hardcoding
dataset-specific logic or adding heavy profiling dependencies.

## Data Source Semantic Notes

Deterministic profiling gives the product raw structure: dtypes, nullability,
sample values, missingness, and simple inferred roles. Semantic notes add
human-written business meaning on top of that structure.

Semantic notes include source-level fields:

- source description
- business context
- global caveats

They also include column-level fields:

- display name
- description
- business meaning
- semantic role
- caveats
- examples

Semantic roles are:

- `metric`
- `dimension`
- `timestamp`
- `identifier`
- `text`
- `target`
- `unknown`

When `DataSourceUsageContext` is built, semantic notes enrich the context sent
to the agent. Source description and business context are merged into the data
source description, global caveats are added to context caveats, and column
semantic roles override deterministic inferred roles. The deterministic role is
kept in the column notes so the override remains understandable.

No LLM-based semantic inference is used yet. The point is to let users teach
the product the business meaning of columns without hardcoding
dataset-specific logic.

## Investigation Runs

Investigation execution is now represented as a first-class backend object.
This is separate from the older embedded execution history stored inside an
Investigation and prepares the product for async jobs, websocket streaming, and
a dedicated frontend.

An `InvestigationRun` stores:

- run id
- Investigation id
- created, started, and completed timestamps
- status
- current stage
- data source ids
- normalized run context summary
- error message
- artifact ids
- report ids
- metadata

Run statuses are:

- `queued`
- `running`
- `completed`
- `failed`
- `cancelled`

## Run lifecycle

The user-facing run stages are:

- `preparing_data`
- `building_context`
- `running_analysis`
- `validating_results`
- `generating_report`
- `completed`

These stages are product-level execution states, not raw agent graph internals.
They intentionally avoid exposing planner/codegen/executor node details in the
UI or API.

During `building_context`, the run service gathers linked data sources,
profiles, semantic notes, and usage contexts into a normalized context summary.
During `running_analysis`, it delegates to the existing `InvestigationService`
and agent runtime. During `validating_results`, it records lightweight checks
such as findings count, artifacts count, report generation, fatal errors, and
empty-result warnings.

## API-driven execution

`POST /investigations/{id}/run` is now the backend entrypoint for running an
Investigation. It accepts `data_source_ids` and `force_refresh_context`, builds
the data context server-side, runs the existing service layer synchronously, and
returns the persisted `InvestigationRun`.

Current execution is still synchronous. There is no Celery, Redis, queue, or
websocket infrastructure yet. The point of this layer is to make execution
backend-centric first, so the same API contract can later be powered by an
async worker model without changing the future frontend.

## Run Events and Stage Timeline

Run events are a user-facing execution timeline for `InvestigationRun`. They
are separate from raw agent diagnostic logs and avoid exposing internal graph node
names.

Event types are:

- `stage_started`
- `stage_completed`
- `info`
- `warning`
- `error`
- `artifact_created`
- `report_created`
- `validation_check`

Events store the run id, Investigation id, event type, stage, message,
timestamp, severity, and compact metadata. The current implementation writes
events during synchronous execution, but the event log is shaped for future
async jobs and streaming UI updates.

The Streamlit Run History shows these events as a simple timeline with
timestamp, severity, stage, and message. Technical stack traces and raw graph
events are not shown in the main timeline.

## Polling-friendly run events

Run event endpoints support incremental polling through a stable cursor:

```text
created_at|event_id
```

Clients can call:

```text
GET /investigation-runs/{run_id}/events?after=<cursor>&limit=100
GET /investigations/{id}/events?after=<cursor>&limit=100
```

Responses use this shape:

```json
{
  "events": [],
  "next_cursor": "2026-05-12T10:00:00+00:00|event_...",
  "has_more": false
}
```

This lets a future Next.js timeline poll for only new events and render a
live-like experience without WebSockets. WebSocket infrastructure, background
workers, Redis, and Celery are intentionally not part of this stage; execution
is still synchronous, but event reads are now streaming-friendly.

## Next.js Decision Workspace shell

A thin Next.js frontend shell lives in:

```text
frontend/
```

It uses the App Router, TypeScript, Tailwind, and a small typed API client in:

```text
frontend/lib/api.ts
```

The shell is intentionally lightweight. It does not replace Streamlit yet and
does not add auth, WebSockets, Redux, a heavy UI kit, or a full frontend
rewrite. Its purpose is to validate that the product backend can support a
dedicated web client.

Current pages:

- `/investigations`
- `/investigations/[id]`
- `/data-sources`
- `/data-sources/[id]`
- `/reports`
- `/reports/[id]`
- `/published-reports`
- `/published-reports/[id]`

The frontend now uses a unified application shell with a persistent sidebar,
top breadcrumb header, active navigation states, recent investigations, and a
theme toggle. The theme system supports `system`, `light`, and `dark`
preferences, stores the preference in `localStorage`, and keeps the report
reader, workspace cards, timelines, forms, and editors readable in both themes.

Lifecycle actions use soft archive flows rather than hard deletion:

- Investigations can be archived from their detail page.
- Data sources can be archived from their detail page.
- Draft ShareableReports can be archived from their report page.

Each destructive lifecycle action is guarded by a confirmation modal. Final
snapshots remain immutable and are not deleted by these archive actions.

## Next.js Investigation Workspace

The Investigation detail page is now the main analytical workspace. It is not a
chat page; conversation is only one interaction layer inside the broader
Investigation object.

The page is organized around:

- a top Investigation header with the current question, status, linked data
  source count, and latest run state;
- a main Investigation canvas for findings, artifacts, and report path
  summaries;
- a sticky side panel with linked data context, follow-up conversation, run
  controls, run history, and selected run timeline.

Data context cards show source name, profile shape, inferred or semantic key
columns, caveats, and links back to Data Source detail pages. This keeps the
agent output grounded in the underlying dataset.

Run timeline remains polling-based and uses:

```text
GET /investigation-runs/{run_id}/events?after=<cursor>&limit=100
```

This makes execution understandable without exposing raw graph internals.
Reports and final deliverables remain downstream outputs after findings and
artifacts are reviewed.

## Report actions in Investigation Workspace

The Next.js Investigation Workspace now exposes report lifecycle actions
directly from the Investigation canvas:

```text
Run analysis
  -> review findings/artifacts
  -> create ShareableReport
  -> view draft report
  -> finalize into FinalReportSnapshot
  -> open final report reader
```

The frontend uses these API routes:

- `GET /investigations/{id}/reports`
- `POST /investigations/{id}/reports`
- `GET /reports/{id}`
- `GET /reports/{id}/readiness`
- `POST /reports/{id}/finalize`
- `GET /reports/{id}/final-snapshots`

The report actions card lets users choose a template, include or exclude
technical appendix content, create a draft ShareableReport, open the readable
draft report page, and create a final snapshot when the report is ready.

Final snapshots are linked from the Reports workflow, so the Investigation
Workspace remains the working area while exported reports remain the deliverable
view.

Run the backend:

```bash
uvicorn source.api.app:app --reload
```

Run the frontend:

```bash
cd frontend
npm install
npm run dev
```

Configure the frontend API target with:

```bash
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

Streamlit remains useful as the temporary demo/workbench client. The Next.js
shell validates the API-first architecture and provides a foundation for a
future polished Decision Workspace.

## Next.js full-cycle run

The Investigation detail page can now trigger the backend execution loop:

```text
Next.js detail page
-> POST /investigations/{id}/run
-> persisted InvestigationRun
-> polling RunTimeline
-> completed or failed run state
```

The run button lives in the thin frontend shell and calls the typed API client
instead of touching backend internals. Linked data sources can be included in
the request, and the request can ask the backend to refresh context. Because
execution is still synchronous, the button shows a simple loading state while
the request is in flight; once the response returns, the new run is inserted at
the top of the local run list.

`RunTimeline` fetches initial events and then polls every few seconds while the
run is `queued` or `running`. It uses the cursor-based event API, deduplicates
events by id, and renders severity, stage, timestamp, and message. Completed
runs still load their timeline once.

This validates the full API-first product loop before introducing async
workers, WebSockets, auth, or a heavier frontend architecture.

## Full workflow from Next.js

The Next.js shell can now cover the core workflow without requiring Streamlit:

```text
Create Data Source
-> Create Investigation
-> Link Data Source
-> Run Investigation
-> View Timeline
```

`/data-sources` includes a metadata-only creation form for source name, type,
location, description, and tags. CSV file upload remains in Streamlit for now;
the shell intentionally validates product object creation before adding upload
plumbing.

`/investigations` includes a compact Start new analysis panel. The user selects
existing data sources, enters a question, creates an Investigation, and is sent
to the Investigation detail page. The detail page can then start a backend run
with the linked data sources and show the polling timeline through the run event
API.

This means Streamlit is no longer required for the primary product loop. It
remains available as a temporary demo/workbench client, while FastAPI is the
source of truth for product state and Next.js acts as a thin API client.

## CSV upload through API

Next.js can now start the workflow from a real CSV file, not only from
metadata-only source records.

Uploaded CSV files are stored locally under:

```text
.analytica/uploads/
```

The upload endpoint is:

```text
POST /data-sources/upload-csv
```

It accepts multipart form data with a `.csv` file plus optional name,
description, and tags. The backend creates a `DataSource`, saves the file as:

```text
.analytica/uploads/{data_source_id}.csv
```

Then it profiles the CSV with the existing pandas-based profiling service and
persists the `DataSourceProfile`. The response includes both the created source
and profile summary, so the Next.js UI can refresh the library and link to the
source detail page immediately.

This is intentionally local-dev storage. It is not S3, object storage,
enterprise file governance, or an async ingestion pipeline. The purpose is to
make the product demo credible: users can upload a real dataset, inspect its
profile, create an Investigation linked to that source, run the Investigation,
and watch the timeline through the existing polling events API.

When a backend Investigation run uses a linked CSV source and no dataframe was
passed directly, the run service loads the dataframe from `DataSource.location`
and passes it into the existing agent runner. Older dataframe-only flows and
Streamlit uploads remain supported.

## Archive vs Delete

Archive is the soft lifecycle action. Archived investigations, data sources,
and draft reports remain available for history and can still preserve links
from existing work.

Delete is permanent. It is exposed as a destructive action with confirmation
in the Next.js UI:

- deleting an investigation removes the investigation, runs, run events,
  messages, findings, artifacts, reports, final snapshots, comments, memory
  items, and related execution history;
- deleting a data source removes the source, profile, semantic notes, and
  references from investigations;
- if the data source is an uploaded CSV stored under `.analytica/uploads/`,
  the local file is deleted as well.

The file deletion path is guarded: the backend only removes files that resolve
inside the managed `.analytica/uploads/` directory. External paths are never
unlinked by the data source delete flow.

## Persistent product layer

Investigations now persist between app restarts. The default database path is:

```bash
.analytica/investigations.sqlite
```

Store selection is controlled by:

```bash
ANALYTICA_INVESTIGATION_STORE=sqlite
ANALYTICA_INVESTIGATION_DB_PATH=.analytica/investigations.sqlite
```

For tests or isolated demos:

```bash
ANALYTICA_INVESTIGATION_STORE=memory
```

SQLite stores investigations, runs, artifacts, findings, reports, and trace
payloads. Complex payloads are serialized as JSON text so the product schema can
evolve without forcing a premature ORM or migration system.

### Artifact lifecycle

Artifacts have a type, visibility, pin state, run id, metadata, and creation
time.

- `user`: tables, charts, reports, and user-facing text.
- `technical`: generated Python, SQL, validation, and run trace.
- `hidden`: reserved for raw/internal payloads.

The Streamlit workspace shows user-facing artifacts by default. Technical
artifacts stay behind an explicit toggle so the main report does not feel like a
diagnostic console.

### Review lifecycle

Investigations support review states such as `needs_review`, `verified`,
`needs_more_analysis`, `failed`, and `archived`.

Findings have their own review status:

- `proposed`
- `accepted`
- `rejected`

The current Streamlit UI exposes simple actions for marking investigations and
accepting/rejecting findings. This is intentionally lightweight, but it creates
the product contract needed for a later collaborative review interface.

### Why Streamlit is still temporary

Streamlit is now only the shell for validating the product model. The durable
objects live in `source/product/*`, not in Streamlit session state. The next
frontend can reuse the same store/service concepts while adding better
navigation, collaboration, report editing, and artifact layout.

## Exporting investigations and reports

The workspace can export an Investigation or ShareableReport as TXT, Markdown,
or HTML.

The default export is a clean business report:

- title, status, question, created/updated timestamps
- structured DecisionReport sections
- accepted findings first, then proposed findings
- user-facing artifacts
- pinned artifacts before unpinned artifacts

TXT is the simplest human-readable format. It uses plain headings, readable
metadata, and numbered or spaced sections without raw JSON.

Markdown is intended for editable/shareable report workflows. It uses a title,
metadata table, section headings, findings, and readable artifact summaries.

HTML is a print-friendly polished report with embedded CSS, readable max width,
metadata badges, section spacing, styled tables, and escaped user content. A PDF
route is intentionally not added yet; users can open the HTML export and use the
browser's Print / Save as PDF flow without adding a heavy PDF dependency.

Technical artifacts such as generated Python, SQL, validation, and trace are not
included by default. The user must explicitly enable the technical appendix in
the Streamlit UI before downloading.

Hidden artifacts are never exported, even with the technical appendix enabled.

Export functions live in `source/product/exporter.py`:

```python
export_investigation_markdown(investigation, include_technical=False)
export_investigation_html(investigation, include_technical=False)
export_shareable_report_txt(report)
export_shareable_report_markdown(report)
export_shareable_report_html(report)
export_final_report_snapshot_txt(snapshot)
build_artifact_summary(investigation, include_technical=False)
```

The HTML export is dependency-free and escapes user content before rendering.

## Shareable reports

`Investigation` and `ShareableReport` are deliberately separate product objects.

- `Investigation` is the working state: runs, trace, raw artifacts, rejected
  findings, drafts, and review decisions.
- `ShareableReport` is the clean deliverable: ordered sections, selected
  findings, selected artifacts, template, version, and export-ready content.

Shareable reports are generated from an Investigation through
`source/product/report_builder.py`.

Available templates:

- `executive_summary`
- `product_decision_memo`
- `technical_appendix`

The executive summary template includes the decision question, answer, key
findings, evidence, limitations, next steps, and selected artifacts. It uses
accepted findings first. If there are no accepted findings, it falls back to
proposed findings. Rejected findings are not included.

The product decision memo template frames the same material as context,
decision question, recommendation, supporting evidence, risks/limitations, next
steps, and artifacts.

The technical appendix template is explicitly for technical readers. It can
include SQL, generated code, validation, trace, and other technical artifacts,
but hidden artifacts are still excluded.

The Streamlit UI has a Reports section for the selected Investigation:

- choose a template
- optionally include technical content
- create a ShareableReport
- preview the final document
- edit report sections
- create versioned human-edited snapshots
- restore earlier report versions
- download Markdown or HTML
- archive the report

This is the bridge from analytics workspace to a user-facing deliverable that
can be sent to a supervisor, team, or manager.

## Editable reports

Shareable reports are now editable documents, not only generated exports.

Each `ReportSection` stores:

- `title`
- `content`
- `order`
- `artifact_ids`
- `edited_by_user`
- `created_by`
- `version`
- `review_status`
- `updated_at`
- `edit_history`

The report editing service supports:

- updating section titles
- updating section content
- reordering sections
- adding sections
- removing sections
- duplicating sections
- adding and resolving lightweight review comments
- moving reports through approval states

Any user edit marks the section as `edited_by_user=True`, increments the
section version, updates timestamps, and records a lightweight edit history
entry. This keeps the data model simple while making human edits explicit.

## Draft report editing in Next.js

The Next.js draft report page at `/reports/[id]` now supports lightweight
human-in-the-loop editing without a rich-text editor.

The page has a simple `Preview / Edit` toggle. In edit mode a user can:

- edit section titles;
- edit section content in plain text areas;
- add a new section;
- duplicate a section;
- delete a section;
- move sections up or down.

The frontend calls thin FastAPI routes over `ReportEditingService`:

- `PATCH /reports/{id}/sections/{section_id}`
- `POST /reports/{id}/sections`
- `DELETE /reports/{id}/sections/{section_id}`
- `POST /reports/{id}/sections/{section_id}/duplicate`
- `POST /reports/{id}/sections/reorder`

The backend keeps the existing report versioning behavior: each edit creates a
new ShareableReport version snapshot and marks edited title/content sections
back to draft review status. The readiness panel on the report page refreshes
after edits through the Next.js server render path.

Final snapshots remain immutable. Editing a draft report after finalization does
not mutate already-published final deliverables.

## Section-level report review in Next.js

The Next.js draft report page also supports lightweight section review controls.
Each section shows its review status and can be marked:

- `approved`
- `changes_requested`

The UI calls the existing backend review routes:

- `POST /reports/{id}/sections/{section_id}/approve`
- `POST /reports/{id}/sections/{section_id}/request-changes`

Editing title or content returns a section to draft status through the existing
backend behavior. Review actions create a new report version snapshot and the
report page refreshes, so readiness counts and blocking checks update after the
change.

The readiness panel shows whether the report is ready, plus blocking checks and
warnings. Finalization still follows backend rules: reports with blocking
readiness issues cannot be finalized unless the user explicitly uses the force
option. This is lightweight review, not full collaboration.

## Section comments in report review

The Next.js draft report page now shows review comments under each report
section. Comments are lightweight annotations scoped to a section, not realtime
collaboration threads.

Reviewers can:

- add a comment explaining what should be clarified;
- request changes with an optional reason, which creates a section comment;
- resolve open comments after edits are made;
- delete comments when they are no longer useful.

Open comments are included in report readiness checks and block finalization by
default. Resolved comments remain visible in the section history when the user
chooses to show them. The API routes are:

- `GET /reports/{id}/comments`
- `POST /reports/{id}/comments`
- `POST /reports/{id}/comments/{comment_id}/resolve`
- `DELETE /reports/{id}/comments/{comment_id}`

## Report review summary

The Next.js draft report page includes a `Report Review` summary in the right
sidebar. It gives reviewers an at-a-glance view of:

- total sections;
- approved sections;
- draft / needs-review sections;
- sections with requested changes;
- open and resolved comments;
- blocking readiness checks;
- warning checks.

The summary derives its counts from the current `ShareableReport`, section
comments, and readiness result. It also shows an overall state such as
`Ready to finalize`, `Needs attention`, `Changes requested`, or `Draft`.

When possible, the panel links to the first section with open comments or
requested changes. This keeps finalization grounded in lightweight review state:
open comments and blocking readiness issues should be resolved before creating
an immutable final snapshot.

## Human-in-the-loop workflow

The intended product workflow is:

```text
AI draft -> human editing -> review comments -> resolved comments -> approved report -> export/share
```

The Investigation remains the analytical workspace. It can contain rough runs,
technical trace, proposed or rejected findings, and intermediate artifacts.

The ShareableReport is the polished document. A human can refine language,
remove irrelevant sections, add context, reorder evidence, and export the final
result without changing the underlying Investigation history.

## Report review workflow

Shareable reports now support lightweight review annotations. This is not full
realtime collaboration: there are no user accounts, presence indicators,
mentions, permissions, or threaded discussions yet. The goal is a small review
contract that fits the current Streamlit product shell.

Each `ReportComment` is attached to one report section and stores:

- `report_id`
- `section_id`
- `text`
- `status`
- `created_at`
- `updated_at`
- `resolved_at`
- `author`
- `metadata`

Comment statuses are:

- `open`
- `resolved`

Shareable reports also have an `approval_status`:

- `draft`
- `in_review`
- `changes_requested`
- `approved`

The Streamlit Reports section allows a reviewer to send a report to review,
request changes with reviewer notes, approve the report, add comments to
individual sections, and resolve comments.

Reports with open comments cannot be approved by default. A service-level
`force=True` escape hatch exists for controlled workflows, but the UI keeps the
normal path conservative.

Exports include approval metadata such as approval status, approver, and
approval timestamp. Review comments are not included by default. They can be
included explicitly as a review appendix, without exposing raw metadata JSON.

## Report readiness checks

Readiness is a lightweight quality gate for ShareableReports. It is not
enterprise governance: there are no policy engines, access rules, data
certification workflows, or mandatory approval chains. The purpose is to help a
user see whether a report is safe to use as a final deliverable.

Section-level review statuses are:

- `draft`
- `needs_review`
- `approved`
- `changes_requested`

Editing a section returns it to `draft`. A reviewer can approve a section or
request changes from the Streamlit report editor.

Readiness checks are generated on demand by:

```python
evaluate_report_readiness(report, investigation=None, comments=None)
```

Blocking checks fail readiness:

- open comments exist
- report has no sections
- any section has empty content
- any section has `changes_requested`
- report approval status is `changes_requested`

Warnings do not block readiness, but they show product risk:

- no accepted findings in the source Investigation
- no valid evidence artifacts are linked
- no sections are approved
- report is not `in_review` or `approved`
- technical appendix is included in a business-facing template

Hidden artifacts never count as valid evidence. Technical artifacts only count
as evidence for the technical appendix template.

Report approval now evaluates readiness first. Approval is blocked when
blocking checks fail, unless the caller explicitly uses `force=True`. Normal
exports are still allowed, but final export should be treated as ready only when
readiness passes.

## Final report mode

Final Report Mode separates draft downloads from official deliverables.

A draft download exports the current ShareableReport version. It is useful for
checking wording, sharing a work-in-progress, or reviewing formatting. It does
not create a durable publishing record.

A final deliverable creates a `FinalReportSnapshot`. This is an immutable
snapshot of the report at the moment of final export. It stores:

- report id
- investigation id
- report version
- title
- created_at
- created_by
- status
- Markdown content
- HTML content
- TXT content
- readiness snapshot
- approval status
- approved_at / approved_by
- source report JSON
- metadata

Final snapshots are created through the report service. By default, a report
must be approved and must pass readiness checks before it can become final.
`force=True` exists for exceptional local workflows, but the normal path keeps
final export conservative.

The snapshot stores generated Markdown and HTML content at creation time. If the
source ShareableReport changes later, the old final snapshot remains unchanged.

Final snapshots can be revoked, but not deleted through the product workflow.
Revoked snapshots stay stored for auditability. This is lightweight publishing,
not enterprise compliance: there are still no users, auth, legal hold, or formal
policy engine.

## Final report snapshots

Final deliverables are backed by immutable `FinalReportSnapshot` records. A
snapshot stores the report title, source Investigation, finalized report
version, approval metadata, readiness snapshot, Markdown content, and HTML
content as they existed at finalization time.

Downloads never regenerate finalized report content. They use the stored
snapshot payload, so final deliverables remain stable even when the source
ShareableReport changes later.

Revoked snapshots remain stored for auditability. This keeps finalization
traceable without adding a separate publishing UI.

## Decision library metadata

Final report content is immutable, but decision metadata is editable. This lets
report exports carry searchable decision context instead of becoming a flat file
dump.

Decision metadata fields:

- `tags`
- `owner`
- `audience`
- `business_area`
- `decision_date`
- `decision_status`
- `short_description`

Decision statuses are:

- `proposed`
- `accepted`
- `rejected`
- `superseded`
- `unknown`

Tags are normalized and deduplicated. Updating metadata does not mutate
`markdown_content`, `html_content`, readiness snapshots, approval metadata, or
source report JSON. Existing final snapshots without metadata load with empty
metadata and `decision_status=unknown`.

## Report versioning

Editing a ShareableReport creates a new report snapshot. Older versions remain
accessible so a user can compare or restore a previous document state.

Report-level version fields:

- `version`
- `previous_version_id`
- `is_latest`
- `version_note`

The latest version is marked automatically. Restoring an old version creates a
new latest snapshot based on that old content, rather than mutating history in
place.

This is intentionally snapshot-based, not a diff engine. It is enough for the
current product layer and keeps the storage implementation easy to migrate to a
dedicated backend later.

## Product API layer

A minimal FastAPI product API now lives under `source/api/`.

Current routes:

- `GET /investigations`
- `POST /investigations`
- `GET /investigations/{id}`
- `POST /investigations/{id}/messages`
- `GET /investigations/{id}/messages`
- `POST /investigations/{id}/run`
- `GET /investigations/{id}/runs`
- `GET /investigations/{id}/events`
- `GET /investigation-runs/{run_id}`
- `GET /investigation-runs/{run_id}/events`
- `GET /reports/{id}`
- `GET /reports/{id}/versions`
- `GET /reports/{id}/readiness`
- `GET /reports/{id}/comments`
- `POST /reports/{id}/comments`
- `POST /reports/{id}/approve`
- `POST /reports/{id}/request-changes`
- `POST /reports/{id}/sections/{section_id}/approve`
- `POST /reports/{id}/sections/{section_id}/request-changes`
- `POST /reports/{id}/finalize`
- `GET /reports/{id}/final-snapshots`
- `GET /reports/final-snapshots/{snapshot_id}`
- `POST /reports/final-snapshots/{snapshot_id}/revoke`
- `GET /final-reports`
- `GET /final-reports/{snapshot_id}`
- `GET /final-reports/{snapshot_id}/download/txt`
- `GET /final-reports/{snapshot_id}/download/markdown`
- `GET /final-reports/{snapshot_id}/download/html`
- `POST /final-reports/{snapshot_id}/revoke`
- `GET /data-sources`
- `POST /data-sources`
- `GET /data-sources/{id}`
- `GET /data-sources/{id}/profile`
- `GET /data-sources/{id}/usage-context`
- `GET /data-sources/{id}/semantic-notes`
- `PUT /data-sources/{id}/semantic-notes`
- `PATCH /data-sources/{id}/semantic-notes/columns/{column_name}`
- `DELETE /data-sources/{id}/semantic-notes/columns/{column_name}`
- `PATCH /data-sources/{id}`
- `POST /data-sources/{id}/archive`

The API returns JSON-serializable product models and uses the same store factory
as the Streamlit workspace. It does not add authentication, background jobs, or
websocket streaming yet.

The purpose is architectural separation: Streamlit can remain a temporary
product shell while the product backend becomes reusable by a future dedicated
web frontend.

## SQLite schema migrations

SQLite schema versioning is stored in:

```text
schema_migrations
```

The table contains:

- `version`
- `name`
- `applied_at`

`SQLiteInvestigationStore` applies missing migrations during initialization and
exposes:

```python
get_schema_version() -> int
```

Migration v1 contains the initial product schema for investigations, runs,
artifacts, findings, reports, and schema migrations. Migration v2 adds
`shareable_reports`. Migration v3 adds ShareableReport versioning fields.
Migration v4 adds review annotations, report comments, and approval metadata.
Migration v5 adds the readiness layer contract and persists section review
status inside report sections.
Migration v6 adds `final_report_snapshots` for immutable final deliverables.
Migration v7 adds editable `decision_metadata_json` on final snapshots.
Migration v8 adds the Data Source Registry tables and persisted Investigation
links through `linked_data_source_ids_json`.
Migration v9 adds `data_source_semantic_notes` for human-written source and
column semantics.
Migration v10 adds `investigation_runs` for API-driven execution state.
Migration v11 adds `investigation_run_events` for the user-facing run timeline.
Migration v12 adds `investigation_messages` for follow-up questions and
assistant run summaries inside an Investigation.
Migration v13 adds `txt_content` to final report snapshots so plain-text final
deliverables are immutable like Markdown and HTML.
Migrations are written to be safe for existing local databases, so reopening an
existing `.analytica/investigations.sqlite` file does not fail.

Future schema changes should be added as new ordered entries in
`source/product/migrations.py`. Keep migrations small, explicit, and safe to run
against an existing `.analytica/investigations.sqlite` file.

## Investigation conversation and follow-ups

An Investigation is now an iterative workspace rather than a single prompt.
The primary product objects remain runs, artifacts, findings, reports, and final
deliverables, but the workspace can also store a compact conversation thread.

`InvestigationMessage` records:

- user follow-up questions;
- assistant analytical answers or compact summaries;
- notes or errors when needed;
- optional `run_id` links when a message is produced by a run.

Follow-up flow:

```text
user adds InvestigationMessage
  -> POST /investigations/{id}/run with message_id
  -> InvestigationRunService builds compact context
  -> agent runs with the follow-up as active question
  -> artifacts/findings/report update the Investigation
  -> assistant answer/summary message is added for the conversation
```

The run context includes the active follow-up, recent messages, linked data
source usage context, previous findings, artifact titles, and the latest report
summary. It intentionally avoids dumping full raw artifacts or every historical
message into the prompt.

API endpoints:

- `POST /investigations/{id}/messages`
- `GET /investigations/{id}/messages`
- `POST /investigations/{id}/run` with optional `message_id`

This is not a chat-only product model. Conversation is one interaction layer
inside the Investigation Workspace; reviewed artifacts and reports remain the
main durable outputs.

## Investigation memory and suggested questions

Investigations now keep lightweight structured memory alongside runs, messages,
artifacts, findings, and reports. Memory items are persisted through the
product store and linked to one Investigation.

Memory types:

- assumptions;
- open questions;
- decisions;
- risks;
- milestones.

Each memory item stores content, status, timestamps, and metadata. The Next.js
Investigation Workspace shows an `Investigation Memory` panel where users can
add items and mark them resolved or archived. This gives follow-up runs durable
context without turning the workspace into a generic chat transcript.

Suggested questions are generated deterministically from data-source profile,
usage context, semantic notes, inferred column roles, numeric metrics,
categorical dimensions, timestamp columns, missingness, and existing findings.
The suggestions remain dataset-agnostic: they never assume fixed business
columns such as revenue, sales, customers, or regions unless those meanings are
provided through semantic notes.

API endpoints:

- `GET /investigations/{id}/memory`
- `POST /investigations/{id}/memory`
- `PATCH /investigations/{id}/memory/{memory_id}`
- `GET /investigations/{id}/suggested-questions`
- `GET /data-sources/{id}/suggested-questions`

The Next.js workspace renders suggestion cards with a `Use as investigation
question` action. That action creates a follow-up message and starts a new run
with the linked data sources, preserving the product loop:

```text
data context -> suggested question -> follow-up run -> findings/artifacts -> memory/report updates
```

## Final report preview in Next.js

The Next.js Reports workspace opens report drafts and finalized snapshots in a
document-style preview. The preview displays:

- report status and decision metadata;
- approval metadata;
- readiness summary;
- source report version;
- export actions.

The Reports page remains the user-facing home for report drafts and final
deliverables.

## Final report reader mode

The final report detail page is a lightweight reader mode. The main canvas is
reserved for the document preview, while a sticky sidebar keeps the report
state visible:

- final/revoked status;
- decision status, owner, audience, business area, tags, and decision date;
- approval metadata;
- readiness counts;
- report version and snapshot id;
- download and print actions.

The preview supports HTML, Markdown, and TXT modes. HTML is rendered inside an
isolated preview frame. Markdown and TXT use readable text containers, and
Markdown headings provide a small section outline without adding a markdown
parser dependency.

The print layout hides navigation, metadata panels, and actions so browser
printing focuses on the report content. Downloads still use immutable
`FinalReportSnapshot` content rather than regenerating the document.

Downloads use the stored final snapshot content:

- TXT for simple readable plain-text sharing;
- Markdown for editable text workflows;
- HTML for polished browser viewing and printing.

Backend PDF export is intentionally skipped for now. The HTML export is
print-friendly, and the Next.js detail page includes a `Print / Save PDF`
action that uses the browser print dialog.

## Promote to memory and evidence linking

Investigation Memory can now be populated from existing analytical work, not
only from manual notes. This keeps the workspace self-organizing while
preserving source references.

Supported promotion flows:

- finding -> assumption;
- finding -> risk;
- finding -> decision;
- report comment -> open question;
- report section -> decision.

Promoted memory items store origin metadata such as `source_type`,
`source_id`, `promoted_from`, and `promoted_at`. The source content is copied
into memory, while the original finding, comment, or report section remains
unchanged.

Findings also support lightweight evidence references. A finding can link to:

- artifacts;
- data sources;
- memory items;
- report sections through the API model.

Evidence links are stored as compact metadata on the finding. This is
intentionally not a graph database: it is a small reference layer that helps a
reviewer understand what supports a finding and navigate back to the relevant
workspace object.

The Next.js Investigation Workspace includes an Evidence Inspector. Clicking a
linked evidence chip opens a side panel instead of navigating away. The
inspector previews the referenced object when it is already available in the
workspace:

- artifact title, type, visibility, and compact content preview;
- data source status, row/column summary, caveats, and link to source detail;
- memory item type, status, content, timestamps, and origin metadata;
- report section title, review status, content preview, and link to report.

This keeps traceability lightweight and readable. It is not a graph database or
a lineage visualization system.

## User-facing display layer

The product can store technical metadata, raw payload previews, generated SQL,
Python code, validation output, and run identifiers. Those details are useful
for diagnostics and audit, but they should not dominate the main Investigation
Workspace.

The Next.js workspace uses a display layer that turns raw analytical payloads
into readable summaries:

- key findings show short explanations instead of raw dictionary/dataframe/SQL
  previews;
- visual analysis highlights charts and report-ready outputs first, with generated
  code, SQL, validation, and traces collapsed under advanced details;
- conversation messages use simple `You` and `Analytica` labels instead of
  internal message types and run ids;
- data context cards show dataset name, shape, key columns, and caveats without
  long inferred-role dumps;
- suggested questions are deduplicated, shortened, and limited so the workspace
  stays readable.

Main UI is for business/user-facing interpretation. Technical UI remains
available through collapsed advanced sections when a reviewer needs
to inspect run ids, raw payloads, SQL, code, trace, or validation internals.

## Conversational Workspace Layout

The Next.js Investigation detail page now uses a center-first conversational
workspace:

```text
Key findings rail | Main chat workspace | Dataset rail
```

The main column is the primary user experience. It contains only the
conversation: user questions, follow-up questions, assistant answers, and
clear analytical answers. Findings are intentionally not rendered as cards
inside the chat. Assistant messages can point the user to key findings or
visual analysis, but the chat remains a readable dialogue rather than a mixed
object feed.

The left rail is compact analytical output:

- key findings;
- visual analysis with chart previews;
- analyst notes only when useful;
- report actions and final deliverable links.

Finding actions also live in this rail. Users can expand a compact finding to
view details, inspect evidence, or use it as report
material without interrupting the central conversation.

The right rail contains supporting context:

- dataset overview;
- suggested next analyses;
- latest chart or report;
- a simple Analyze action with advanced settings collapsed.

Backend stages, polling, run identifiers, validation traces, and diagnostic payloads
do not appear in the default workspace.

## Conversation vs Findings separation

The Investigation Workspace follows a strict separation:

- Center chat: user questions and assistant answers only.
- Left rail: key findings, visual analysis, analyst notes when present, report
  actions, and evidence controls.
- Right rail: dataset overview, suggested next analyses, latest output, and a
  simple Analyze action.

This keeps the product understandable for non-technical users. A failed run is
shown in the chat as a plain assistant error with a useful reason, while raw
stack traces, run ids, SQL, code, and payload metadata stay in collapsed
advanced sections.

The layout is responsive: desktop uses three columns, while smaller screens put
the chat first and stack the knowledge/context rails below it. Lightweight toast
notifications give feedback for actions such as starting analysis, finishing
analysis, archiving, deleting, and other confirmed operations.

## Analytical continuity positioning

Analytica is positioned as an AI analytical investigation agent, not a generic
chat-with-CSV interface. Follow-up questions are treated as continuations of an
ongoing investigation, but the assistant must not describe that machinery to the
user. The backend prompt now explicitly tells the agent to:

- answer the current question first;
- avoid replaying the dataset overview unless asked;
- resolve references such as "this", "that", or "the chart" from recent context;
- use prior conversation, saved insights, outputs, reports, and research notes
  only when they help the current analytical question;
- never explain orchestration, memory, context handling, run lifecycle, backend
  workflow, or prompt mechanics in a user-facing answer.

When the raw CSV is not available to the runtime, deterministic fallbacks also
prefer continuity: questions about evidence review use current insights, and
other follow-ups produce an analyst-facing answer instead of repeating schema
profiling or explaining the continuation mechanism.

Analytica should sound like a senior data analyst. Good answers include
comparisons, rankings, anomalies, caveats, chart suggestions, and concrete next
analytical actions. Bad answers narrate workflow mechanics such as "I treated
this as a continuation" or "I used prior context."

The deterministic fallback layer now covers common analytical intents when the
LLM or tool path fails:

- numeric metric by categorical group;
- outlier detection;
- data quality checks for missing values and duplicates;
- correlations between numeric fields;
- time trends when a date-like column exists;
- chart requests using the best available metric and time/group column;
- evidence review for weak insights.

User-facing language now emphasizes analyst outcomes:

- "Findings" are presented as key findings.
- "Artifacts" are presented as visual analysis.
- Notes are shown as analyst notes only when useful.
- Dataset upload is the primary flow; metadata-only dataset setup is advanced.
- Run, timeline, lifecycle, and orchestration language is hidden from the
  primary workspace.

The product should expose analytical thinking, conclusions, charts, trust, and
report-ready narrative. Internal execution concepts remain available only as
advanced details.

API endpoints:

- `POST /investigations/{id}/findings/{finding_id}/promote-memory`
- `POST /investigations/{id}/findings/{finding_id}/evidence`
- `POST /reports/{report_id}/comments/{comment_id}/promote-memory`
- `POST /reports/{report_id}/sections/{section_id}/promote-memory`

## Organizational analytical workflows

Analytica now includes a lightweight organizational guidance layer. It is not a
task-management or approval system; it gives analysts reusable investigation
standards without forcing rigid process.

The backend models:

- organizational workflows, made of stages and expectations;
- validation expectations and evidence requirements;
- lightweight investigation review feedback;
- organizational playbooks derived from reusable analytical patterns;
- report standards for executive briefs, decision memos, and technical
  appendices.

The guidance stays dataset-agnostic. It works from semantic analysis types,
artifacts, evidence strength, and reusable cross-investigation patterns rather
than specific column names or sample datasets. For example, a grouped metric
finding should have chart/table evidence, anomaly claims should include
magnitude or validation guidance, and correlation findings should avoid causal
overclaiming.

The Next.js workspace shows this as compact analytical guidance in the right
rail: suggested workflow, evidence quality, next review checkpoint, and a small
number of validation hints. This is progressive guidance for analysts, not
enterprise workflow bureaucracy.

API endpoint:

- `GET /investigations/{id}/workflow-guidance`

## Workspace dashboard

The Next.js home page is now an analytical dashboard instead of a redirect.
It gives a product-level overview of the workspace:

- recent investigations;
- investigations needing attention;
- active risks from Investigation Memory;
- pending report reviews;
- recent uploaded datasets;
- latest finalized reports;
- suggested next actions.

The dashboard is assembled from existing API resources, so it does not add a
new orchestration layer. It reinforces the product model:

```text
datasets -> investigations -> memory/runs/evidence -> reports -> final deliverables
```

## Next steps

- Continue polishing the Next.js Investigation Workspace while keeping Streamlit
  as a reference/demo client.
- Add richer report export templates and move from SQLite to Postgres when collaboration requires it.
- Add artifact files/object storage for charts, tables, and exported reports.
- Add real users, auth, section ownership, and collaborative review once the
  dedicated frontend exists.
- Add scheduled monitors that can turn reviewed Investigations into recurring
  checks.
- Add governed metric definitions and data-source permissions.
