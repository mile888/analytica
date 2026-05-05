---
name: reporting
description: Use this skill only at the final response stage after tools have already produced facts or a chart.
allowed-tools:
---

# Reporting Skill

## Purpose
This skill defines how to write the final response to the user.

## Workflow
1. Read the user question.
2. Read only the facts returned by the tools already called in this run.
3. Write a concise answer in Russian.
4. Mention key numbers, categories, or chart facts when available.
5. Do not include code unless the user asks for it.

## Tools to Use
- Do not call a separate reporter tool.
- Write the final answer directly from the facts returned by `top_n`, `plot_bar`, `run_bar_command`, or `run_python_analysis`.
- For business-only answers, use `business-analysis`; do not use this skill to decide whether data tools are needed.

## Rules
- Do not hallucinate results.
- Do not mention analysis steps that were not actually performed.
- Do not over-explain.
- If there was an execution error, explain it briefly and clearly.
- Keep the response professional and easy to read.
