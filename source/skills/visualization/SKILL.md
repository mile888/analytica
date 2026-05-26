---
name: visualization
description: Chart and plot creation from current dataset columns, including bar, grouped, stacked, share, and custom matplotlib visualizations.
---

# Visualization

## When To Use
Use this skill when the user asks for a chart, plot, visual comparison, distribution, ranking visualization, or `/bar` command.

## Workflow
1. Inspect schema before choosing chart columns.
2. Verify requested columns exist.
3. Aggregate before plotting when raw rows are too granular.
4. Prefer `plot_bar` or `run_bar_command` for supported bar charts.
5. Use `run_python_analysis` only for custom matplotlib charts.
6. Return a matplotlib figure in `result`.

## Rules
- Do not plot invented data.
- Use readable titles and axis labels.
- Rotate category labels when needed.
- If fewer categories exist than requested top N, mention that in the final answer.
