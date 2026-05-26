---
name: data-analysis
description: General workflow for answering questions from the current dataset using schema-first analysis and tool-grounded facts.
---

# Data Analysis

## When To Use
Use this skill when the user asks for calculations, comparisons, summaries, anomalies, rankings, or explanations grounded in the current dataset.

## Workflow
1. Inspect the dataset schema before selecting columns.
2. Use only real columns returned by `inspect_dataset_schema`.
3. Prefer the narrowest safe tool: `top_n` for grouped rankings, `find_drops` for date-based drops, `run_python_analysis` for custom analysis.
4. If code is needed, assign the final value to `result`.
5. If execution fails, use the error message to fix tool arguments or code once before explaining the limitation.
6. Base the final answer only on tool output or user-provided facts.

## Rules
- Do not invent data, columns, metrics, filters, or scenarios.
- Do not assume a sample dataset shape.
- If the dataset lacks required columns, say what is missing.
- Use `write_report_artifact` only when the user asks for a report/file or the answer is long enough to preserve as an artifact.
