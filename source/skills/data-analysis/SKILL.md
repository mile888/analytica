---
name: data-analysis
description: Use this skill when the user asks to analyze a dataset, calculate metrics, aggregate columns, compare groups, find top values, inspect schema, or answer a question using table data.
allowed-tools: inspect_dataset_schema, top_n, find_sales_drops, run_python_analysis
---

# Data Analysis Skill

## Purpose
This skill guides the agent through structured dataset analysis.

## Workflow
1. Decide whether the user request requires data.
2. Inspect the dataset schema before writing code.
3. Use the most specific computation tool available.
4. For top/ranking questions, call `top_n`.
5. Use `run_python_analysis` only when `top_n` is not enough.
6. For anomaly/drop questions by date, call `find_sales_drops`.
7. If execution fails, fix the tool arguments or code using the error message.
8. Return only facts supported by the computed result.

## Tools to Use
- Use `inspect_dataset_schema` before choosing columns.
- Use `top_n` for top-N grouped aggregation.
- Use `find_sales_drops` for daily anomaly/drop questions.
- Use `run_python_analysis` for custom calculations that cannot be expressed with `top_n`.
- Do not use an extra reporter/codegen LLM tool; the Deep Agent writes the final answer itself.

## Rules
- Do not invent columns.
- Do not invent results.
- Always use the available schema.
- The final answer must be based on the actual execution output.
- If the request cannot be answered from the dataset, say what is missing.
