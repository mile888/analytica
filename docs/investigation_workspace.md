# Investigation Workspace

The investigation workspace is the product surface for moving from uploaded datasets to executable analysis, artifacts, findings, and reports.

## Product Model

```text
dataset upload -> investigation -> runs -> artifacts/findings -> report -> export
```

Core objects:

- `Investigation`: durable analytical workspace for a question or thread of questions.
- `InvestigationRun`: one execution attempt against resolved dataset scope.
- `Artifact`: generated chart, table, SQL, code, text result, validation trace, or report evidence.
- `Finding`: reviewable insight, limitation, or conclusion.
- `ShareableReport`: persisted structured report built from history, findings, limitations, branches, and selected artifacts.
- `FinalReportSnapshot`: immutable published report state.

## DeepAgents Context

The agent uses the official DeepAgents runtime with typed `AnalyticaContext`. Tools that need run-specific state read it from injected runtime context instead of globals or prompt-only state.

Memory and artifacts are exposed through DeepAgents filesystem backends:

- `/memories/` maps to `.analytica/memory/`
- `/artifacts/` maps to the configured artifact directory
- `/source/skills/` exposes checked-in skill folders

Long artifacts should be written to filesystem-backed artifacts instead of repeatedly inserted into prompts.

## Dataset-Aware Continuity

Each investigation maintains a dataset registry. The resolver selects a dataset scope using explicit names, schema matches, semantic roles, sample values, branch/artifact lineage, recent conversation context, user selection, and runtime availability.

Follow-up analysis receives the latest branch and artifact lineage so questions such as “remove outliers” or “explain the adjusted chart” stay attached to the correct dataset and chart unless the user explicitly switches context.

## Branches And Artifacts

`AuthoritativeExecutionPlanner` creates structured plans with metric, dimension, filters, transformations, chart intent, and validation requirements. `BranchWorkspaceManager` stores branch metadata, active artifacts, dataset scope, and continuation context.

Artifacts carry enough metadata for report inclusion:

- dataset lineage;
- branch lineage;
- metric, dimension, filters, and chart type;
- originating question;
- persisted static image reference when selected for reports.

## Reports

Reports are built as structured objects, not one-shot summaries. The builder ingests:

- user questions and assistant answers;
- findings and conclusions;
- limitations and warnings;
- selected chart/table artifacts;
- analytical branches;
- dataset references;
- timestamps and ordering.

The PDF exporter embeds stored chart images directly, preserving chart order and captions without regenerating visuals during export.

## Frontend

The Next.js workspace provides:

- dataset upload and preview;
- investigation chat;
- branch switcher and active dataset labels;
- chart/table artifact cards;
- finding review surfaces;
- report creation, preview, visual analysis, and PDF export.

The frontend stays thin: durable product state is fetched from the FastAPI backend and persisted through product APIs.
