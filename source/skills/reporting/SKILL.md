---
name: reporting
description: Final analytical response and optional markdown report artifact creation from facts already produced by tools.
allowed-tools: write_report_artifact
---

# Reporting

## When To Use
Use this skill after analysis tools have produced facts, previews, SQL results, code results, or plots.

## Workflow
1. Read the user question and the tool outputs from this run.
2. Summarize the answer in Russian with the smallest useful structure.
3. Mention key numbers, categories, limitations, chart facts, or artifact paths when relevant.
4. If the user requested a reusable report, call `write_report_artifact`.

## Rules
- Do not introduce new unsupported facts.
- Do not describe steps that did not happen.
- Do not include generated code unless the user asks for it.
- Explain execution errors briefly and concretely.
