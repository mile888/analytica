---
name: csv-dataframe-analysis
description: CSV loading assumptions and pandas DataFrame workflow for schema inspection, profiling, column-safe transformations, and exploratory analysis.
---

# CSV DataFrame Analysis

## When To Use
Use this skill when the user asks about CSV data, DataFrame structure, column types, missing values, transformations, or exploratory analysis over the uploaded/current table.

## Workflow
1. Treat the current table as the source of truth. Do not load another file from generated code.
2. Inspect schema/profile first.
3. Use inferred numeric, categorical, and date-like columns only as suggestions; verify names before using them.
4. Convert columns with `pd.to_numeric` or `pd.to_datetime` inside analysis code only when needed.
5. Keep previews bounded and return compact DataFrames or scalar summaries.

## Rules
- Do not hardcode paths, dataset names, or column names.
- Do not call `pd.read_csv` in generated code; CSV loading is handled by entrypoints/tools outside executor.
- Do not mutate or export user data from generated analysis code.
