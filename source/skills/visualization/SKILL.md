---
name: visualization
description: Use this skill when the user asks to create a chart, plot, bar chart, grouped chart, stacked chart, share chart, or visual summary from dataset columns.
allowed-tools: inspect_dataset_schema, plot_bar, run_bar_command, run_python_analysis
---

# Visualization Skill

## Purpose
This skill guides chart creation for analytical questions.

## Workflow
1. Identify the requested chart type.
2. Check that the required columns exist.
3. Aggregate data if needed.
4. Create a readable matplotlib figure.
5. Return the figure as the result.
6. In the final answer, describe only what is visible in the chart.

## Tools to Use
- Use `inspect_dataset_schema` to verify chart columns before code generation.
- Use `plot_bar` for normal bar charts.
- Use `run_bar_command` for `/bar`, `/barh`, `/bar_share`, `/bar_stacked`, and `/bar_grouped`.
- Use `run_python_analysis` only for custom visualizations that cannot be expressed with `plot_bar`.
- Do not use an extra reporter/codegen LLM tool; the Deep Agent writes the final answer itself.

## Rules
- Do not create charts from invented data.
- Use clear axis labels and title.
- Rotate category labels if needed.
- If top=N is requested but fewer categories exist, explain it.
- For `/bar`, `/barh`, `/bar_share`, `/bar_stacked`, `/bar_grouped`, the result must be a plot.
