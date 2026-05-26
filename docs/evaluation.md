# Analytica - Evaluation Summary

This page gives a short summary of the current evaluation evidence for Analytica.
The main benchmark is stored in:
[`notebooks/evaluation_metrics.ipynb`](../notebooks/evaluation_metrics.ipynb).

The audited benchmark outputs are stored in:

```text
docs/evaluation_outputs/
```

## Method

The evaluation uses three rules:

1. **Run real code** - tests use real dataframes and product paths.
2. **Check deterministic results** - expected values are computed with pandas before validation.
3. **Measure several skills** - the benchmark covers single-dataset analysis, multi-dataset analysis, charts, refusals, KPI reasoning, and follow-up correction.

The benchmark evaluates the deterministic product fallback path. Live LLM execution is disabled in this benchmark run. This makes the results reproducible, but it also means that the results do not measure live provider behavior.

## Benchmark Scope

The notebook benchmark contains **19 benchmark cases**.

The cases cover:

- grouped aggregation;
- trend analysis;
- business KPI analysis;
- relationship analysis;
- outlier analysis;
- health-risk analysis;
- visualization requests;
- follow-up correction;
- negative incompatible requests;
- multi-dataset comparison.

The benchmark is small. It should be treated as focused evidence for selected analytical scenarios, not as a complete proof of system reliability.

## Current Metrics

The following values come from `docs/evaluation_outputs/summary_metrics.csv`.

| Metric | Result |
|--------|--------|
| End-to-End Success Rate | 100.0% (19/19) |
| Analytical Correctness | 84.2% (16/19) |
| Semantic Intent Accuracy | 89.5% (17/19) |
| Metric Selection Accuracy | 100.0% (19/19) |
| Aggregation Accuracy | 100.0% (19/19) |
| KPI Correctness | 100.0% (7/7) |
| Groundedness | 100.0% (19/19) |
| Visualization Correctness | 100.0% (11/11) |
| Follow-Up Consistency | 33.3% (1/3) |
| Multi-Dataset Completeness | 100.0% (3/3) |
| Retry Fraction | 0.0% (0/19) |
| Average Latency | 0.043s mean, 0.019s median, 0.381s max |

## Backend Test Result

The audited backend test command required `pypdf` as a transient dependency because this package is used by report-redesign tests but is not installed in the current project environment by default.

The verified command result was:

```text
uv run --with pypdf pytest -q -m "not integration"
```

Result:

```text
1034 passed, 2 failed, 21 deselected
```

The two failed tests are related to healthcare semantic matching. In both failures, the system selected `hospital` instead of `diagnosis` for a question about the most common diagnoses.

## Category Results

The following values come from `docs/evaluation_outputs/category_breakdown.csv`.

| Category | Cases | Analytical Correctness |
|----------|------:|-----------------------:|
| Basic grouped aggregation | 2 | 100.0% |
| Business KPI analysis | 3 | 100.0% |
| Follow-up correction | 3 | 33.3% |
| Health risk analysis | 2 | 50.0% |
| Multi-dataset comparison | 3 | 100.0% |
| Negative incompatible request | 1 | 100.0% |
| Outlier analysis | 1 | 100.0% |
| Relationship analysis | 2 | 100.0% |
| Trend analysis | 1 | 100.0% |
| Visualization request | 1 | 100.0% |

## Failure Analysis

The benchmark has three analytically incorrect cases.

### 1. `health_smoker_prevalence`

Question:

```text
Compare heart disease prevalence between smokers and non-smokers.
```

Expected behavior:

- use `Heart Disease` as the target variable;
- group only by `Smoker`;
- compute prevalence for smokers and non-smokers.

Observed behavior:

- the system used `Heart Disease` as the target variable;
- the system added an extra grouping dimension, `Gender`;
- the output compared smoker and gender combinations instead of smoker-only groups.

Failure type:

```text
grouping, numeric evidence
```

This is a real semantic failure. The system produced useful evidence, but it did not match the requested grouping level.

### 2. `followup_average_to_total_sales`

Question:

```text
Top cities by average Sales. Do not use average, use total Sales.
```

Expected behavior:

- preserve `City` as the grouping field;
- preserve `Sales` as the metric;
- use total sales;
- avoid average sales.

Observed behavior:

- the system preserved `City`;
- the system preserved `Sales`;
- the system still used mean aggregation;
- the answer described average sales.

Failure type:

```text
intent, follow-up
```

This is a real follow-up correction failure. The explicit correction was not applied correctly.

### 3. `followup_filter_total_sales_above_5000`

Question:

```text
Now show only cities with total Sales above 5000.
```

Expected behavior:

- preserve `City` as the grouping field;
- preserve `Sales` as the metric;
- use total sales;
- return only cities where total sales are above 5000.

Observed behavior:

- the system preserved `City`;
- the system preserved `Sales`;
- the system used total sales;
- the system still returned `Berlin`, whose total sales are below the threshold.

Failure type:

```text
intent, numeric evidence, follow-up
```

This is a real follow-up refinement failure. The system handled the main aggregation but did not apply the threshold correctly.

### Passed follow-up case

The expanded benchmark also includes `followup_top3_profit_categories`.
This case passed. It asked the system to narrow a previous profit-by-category analysis to the top three categories with the metric stated explicitly in the follow-up prompt.

The follow-up result is therefore partial. The system passed one simpler top-N refinement, but it failed the aggregation overwrite case and the threshold-filter refinement case.

## Multi-Dataset Evaluation Caveat

The multi-dataset benchmark cases preselect dataset IDs before execution. Therefore, these cases test branch execution, branch evidence packages, and comparative output completeness. They do not fully test dataset resolver accuracy.

The reported multi-dataset completeness value is:

```text
100.0% (3/3)
```

This value should be used only for the tested branch-completeness behavior.

## Groundedness Caveat

Groundedness is checked with deterministic and lexical rules. The benchmark checks that answers do not use unrelated domain terms and that outputs are tied to expected fields and artifacts. This is useful, but it is not a formal proof that every explanation is fully grounded.

## Latency Caveat

Latency values were measured in the local deterministic fallback path.

The audited values are:

```text
mean: 0.043s
median: 0.019s
max: 0.381s
```

These numbers do not include live LLM provider latency.

## Current Limitations

Known limitations:

1. The benchmark has only 19 cases.
2. Live LLM behavior is not evaluated in this benchmark.
3. Multi-dataset benchmark cases preselect dataset IDs.
4. Groundedness checks are partly lexical.
5. Follow-up behavior is partial: one simpler top-N follow-up passed, while two follow-up cases failed.
6. Healthcare semantic matching still has known failures.
7. Frontend tests were not run in the audited environment because `npm` was not available.
8. UI screenshots were not generated in the audited environment.

These limitations should be included in the final report. They do not invalidate the benchmark, but they define its scope.

## Reproducible Outputs

The current audited outputs are:

```text
docs/evaluation_outputs/summary_metrics.csv
docs/evaluation_outputs/benchmark_results.csv
docs/evaluation_outputs/failure_analysis.csv
docs/evaluation_outputs/category_breakdown.csv
docs/evaluation_outputs/benchmark_run_metadata.json
docs/evaluation_outputs/benchmark_metric_dashboard.png
docs/evaluation_outputs/category_accuracy.png
docs/evaluation_outputs/latency_per_case.png
```

For formulas, benchmark definitions, and validation logic, see:

```text
notebooks/evaluation_metrics.ipynb
```
