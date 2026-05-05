---
name: business-analysis
description: Use this skill when the user asks for business interpretation, recommendations, hypotheses, metric design, project framing, or analytical strategy without requiring direct calculations from a dataset.
allowed-tools:
---

# Business Analysis Skill

## Purpose
This skill helps answer business analytics questions without executing code.

## Workflow
1. Identify the business problem.
2. Clarify the goal, metric, or decision if needed.
3. Provide a concise structured answer.
4. Suggest useful metrics, segments, or next analysis steps.
5. Avoid pretending that calculations were performed.

## Tools to Use
- Do not call a separate business-answer tool.
- Answer directly using the Deep Agent model.
- Do not call `inspect_dataset_schema`, `top_n`, `plot_bar`, or `run_python_analysis` for this skill.
- If the user asks for calculations from a dataset, switch to `data-analysis` instead of answering from this skill.

## Rules
- Do not use code.
- Do not claim that data was analyzed if no data execution happened.
- Keep the answer practical and business-oriented.
- Prefer short structured recommendations.
