# Investigation Workspace

Analytica is gaining a product layer centered on an `Investigation`, not a chat
session. An Investigation is a user-owned analytical workspace for moving from a
business question to reviewed artifacts and a DecisionReport.

## Product model

The core lifecycle is:

```text
question -> investigation -> artifacts -> report -> review
```

An Investigation represents an analytical question such as:

- "Как режим доставки связан с объемом продаж?"
- "Какие сегменты клиентов дают наибольшие продажи?"
- "В каких регионах продажи выше или ниже среднего?"

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
    skills=["/source/skills/"],
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
- skills are exposed through the filesystem backend at `/source/skills/`
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

`InvestigationService` runs the existing agent through dependency injection. The
default runner calls the current `source.agent.run_once` function. Agent output
is translated by `agent_output_to_investigation_update`, which maps summaries,
findings, generated code, SQL metadata, artifacts, limitations, and tool
timeline into product objects.

## Current implementation

- `source/product/investigation.py`: product domain models.
- `source/product/store.py`: in-memory `InvestigationStore`.
- `source/product/sqlite_store.py`: persistent SQLite store.
- `source/product/store_factory.py`: store backend selection from environment.
- `source/product/migrations.py`: SQLite schema migrations.
- `source/product/exporter.py`: Markdown and HTML export helpers.
- `source/product/data_sources.py`: Data Source Registry domain models.
- `source/product/data_profiling.py`: lightweight pandas profiling for data
  sources.
- `source/product/report_builder.py`: ShareableReport generation from
  Investigations.
- `source/product/report_service.py`: editable ShareableReport workflow and
  version snapshots.
- `source/product/readiness.py`: lightweight report readiness checks.
- `source/product/final_report_registry.py`: Published Reports Library listing
  and summaries.
- `source/product/adapter.py`: robust adapter from agent output to product
  artifacts, findings, report, and trace.
- `source/product/service.py`: service layer connecting the store to the agent
  runner.
- `source/api/*`: minimal product API foundation for investigations and
  reports.
- `pages/Investigation_Workspace.py`: temporary Streamlit workspace UI.
- `pages/Data_Sources.py`: temporary Streamlit Data Source Registry UI.
- `pages/Published_Reports.py`: Published Reports Library UI.

The in-memory store remains available for tests and fallback. The app defaults
to SQLite persistence.

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
are separate from raw agent/debug logs and avoid exposing internal graph node
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
- `/published-reports`

The Investigation detail page is the first Decision Workspace prototype. It
shows Investigation status, linked data sources, backend runs, findings,
artifacts, and a polling timeline backed by:

```text
GET /investigation-runs/{run_id}/events?after=<cursor>&limit=100
```

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
debug console.

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

## Exporting investigations

The workspace can export an Investigation as Markdown or HTML.

The default export is a clean business report:

- title, status, question, created/updated timestamps
- structured DecisionReport sections
- accepted findings first, then proposed findings
- user-facing artifacts
- pinned artifacts before unpinned artifacts

Technical artifacts such as generated Python, SQL, validation, and trace are not
included by default. The user must explicitly enable the technical appendix in
the Streamlit UI before downloading.

Hidden artifacts are never exported, even with the technical appendix enabled.

Export functions live in `source/product/exporter.py`:

```python
export_investigation_markdown(investigation, include_technical=False)
export_investigation_html(investigation, include_technical=False)
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

## Published reports library

The Published Reports Library is the product-level list of final deliverables.
It is separate from an individual Investigation workspace.

The library is backed by immutable `FinalReportSnapshot` records. It shows what
has been published, when it was published, which Investigation it came from,
which report version was finalized, whether the snapshot is still final or has
been revoked, and whether the readiness snapshot passed when it was created.

The Streamlit page lives at:

```text
pages/Published_Reports.py
```

The page supports:

- search by report title or Investigation title
- search by tags, owner, audience, business area, short description, and
  decision status
- status filtering for `all`, `final`, and `revoked`
- decision status, business area, and tag filtering
- optional Investigation filtering
- Markdown and HTML downloads from stored snapshot content
- snapshot revoke actions
- detail expanders with approval metadata, readiness summary, report version,
  and snapshot id
- metadata editing for the decision library

Downloads never regenerate report content. They use `markdown_content` and
`html_content` stored on the snapshot, so final deliverables remain stable even
when the source ShareableReport changes later.

Revoked snapshots remain visible. This keeps the library auditable without
turning it into a heavy compliance system.

## Decision library metadata

Final report content is immutable, but decision metadata is editable. This lets
the Published Reports Library become a searchable decision library instead of a
flat list of exported files.

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
Migrations are written to be safe for existing local databases, so reopening an
existing `.analytica/investigations.sqlite` file does not fail.

Future schema changes should be added as new ordered entries in
`source/product/migrations.py`. Keep migrations small, explicit, and safe to run
against an existing `.analytica/investigations.sqlite` file.

## Next steps

- Replace the Streamlit shell with a dedicated web app Investigation Workspace.
- Add richer report export templates and move from SQLite to Postgres when collaboration requires it.
- Add artifact files/object storage for charts, tables, and exported reports.
- Add real users, auth, section ownership, and collaborative review once the
  dedicated frontend exists.
- Add scheduled monitors that can turn reviewed Investigations into recurring
  checks.
- Add governed metric definitions and data-source permissions.
