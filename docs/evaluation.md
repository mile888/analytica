# Analytica - Evaluation Summary

This page gives a short summary of the evaluation notebook:
[`notebooks/evaluation_metrics.ipynb`](../notebooks/evaluation_metrics.ipynb).

## Method

The evaluation uses three rules:

1. **Run real code** - tests use real dataframes and product paths.
2. **Check deterministic results** - pandas and SQL results are checked directly.
3. **Measure several skills** - single-dataset, multi-dataset, charts, refusals, and follow-ups are tested.

## Metrics Overview

### 1. End-To-End Success

This checks how many benchmark questions finish without errors.

- Current local validation: **1020 backend tests pass** with non-integration tests.
- Covers: single-dataset analysis, multi-dataset analysis, edge cases, and follow-ups.

### 2. Analytical Accuracy

This checks whether the system uses the right columns, roles, and dataset.

| Dimension | Accuracy | What It Tests |
|-----------|----------|---------------|
| Semantic Role Classification | 100% | Column type detection (metric, dimension, timestamp, identifier) |
| Metric Subtype Classification | 100% | Revenue vs. count vs. ratio classification |
| Dataset Purpose Inference | 93.3% | Automatic dataset role detection (sales, HR, marketing, etc.) |
| Dataset Resolution | 85.7% | Correct dataset selection for ambiguous queries |
| Overall Composite | **95.6%** | Weighted average across all dimensions |

### 3. Retry Fraction

This checks whether the system needs to retry failed generated code.

- **Result: 0%** in the deterministic path.
- The main pipeline does not depend on generated Python code for statistics.

### 4. Average Latency

This measures time from question to answer.

| Pipeline Type | Latency |
|--------------|---------|
| Deterministic analysis | ~200ms |
| LLM-assisted reasoning | ~2.5s |

### 5. Visualization Success Rate

This checks whether chart artifacts have valid data and chart fields.

- **Result: 100%** across tested bar, histogram, line, and scatter charts.
- Each chart includes chart type, rows or bins, axes, and title.

### 6. Multi-Dataset Reasoning

This checks whether the system can reason over more than one dataset.

| Capability | Accuracy |
|-----------|----------|
| Joinability Reasoning | 100% |
| Dataset Role Classification | 86.7% |
| Follow-up Continuation Detection | 100% |
| Strategic Insight Quality | 86.7% |
| **Composite** | **93.3%** |

## Benchmark Query Suite

The notebook benchmark uses 42 questions:

- **Single dataset**: rankings, charts, trends, correlations, and transformations.
- **Multi dataset**: joinability, entity alignment, and comparison.
- **Edge cases**: missing fields, bad filters, unclear inputs, and refusals.

## Failure Analysis

Known limits:

1. **Generic datasets** can have weak role labels.
2. **Ambiguous questions** may need clarification.
3. **LLM wording** can vary, but computed results should stay grounded.

## Strengths

| Strength | Evidence |
|----------|----------|
| Strong deterministic test pass rate | 1020 non-integration backend tests passed locally |
| Zero retry overhead | Deterministic pipeline, no code generation |
| Complete artifact coverage | Every analysis produces valid charts |
| Robust refusal behavior | Invalid queries receive explicit limitations, not fabricated results |
| Bilingual support | English and Russian queries handled identically |
| Multi-dataset reasoning | Cross-dataset analysis without manual schema mapping |

## Full Evaluation

For formulas, benchmark definitions, and charts, see:

```
notebooks/evaluation_metrics.ipynb
```
