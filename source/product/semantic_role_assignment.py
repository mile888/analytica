"""Generic Semantic Role Assignment Engine.

Infers what role each column plays in a user's analytical question:
- target/outcome variable (binary flags, status fields)
- grouping variable (categorical splits: "by X", "between X")
- explanatory/risk-factor variables (predictors in association analysis)
- metric variable (explicit numeric aggregation target)
- temporal variable

This module is entirely generic — no domain-specific column names,
no dataset-specific logic. It detects roles via:
1. Binary column structure (0/1, Yes/No)
2. Outcome-like column name markers
3. Relational question patterns ("X between Y", "X by Y")
4. Operation classification (prevalence, association, comparison)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


# ── Data structures ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class SemanticRoleAssignment:
    """Result of semantic role assignment for a question + dataset pair."""

    target_variable: str | None = None
    grouping_variables: list[str] = field(default_factory=list)
    explanatory_variables: list[str] = field(default_factory=list)
    metric_variable: str | None = None
    time_variable: str | None = None
    operation: str = ""  # prevalence, comparison, association, trend, visualization, ranking
    confidence: float = 0.0
    reasoning: str = ""


# ── Public API ────────────────────────────────────────────────────────────

def assign_semantic_roles(
    question: str,
    df: pd.DataFrame,
    *,
    operation: str | None = None,
) -> SemanticRoleAssignment | None:
    """Main entry point: assign semantic roles to columns for a question.

    Returns None if no meaningful role assignment can be inferred
    (e.g. generic questions like "summarize this dataset").
    """
    if not isinstance(df, pd.DataFrame) or df.empty or not question:
        return None

    normalized = _normalize(question)

    # 1. Classify the operation
    op = operation or _classify_analysis_operation(normalized)
    if not op:
        return None

    # 2. Detect target/outcome variable
    target = _detect_target_variable(normalized, df)

    # 3. Detect grouping variables
    groups = _detect_grouping_variables(normalized, df, exclude={target} if target else set())

    # 4. For association questions, detect explanatory variables
    explanatory: list[str] = []
    if op == "association" and target:
        explanatory = _detect_explanatory_variables(
            normalized, df, target=target, groups=groups,
        )

    # 5. Detect time variable
    time_var = _detect_time_variable(normalized, df)

    # 6. Validate and score
    if not target and op in ("prevalence", "association"):
        return None  # These operations require a target

    if not target and not groups:
        return None  # No meaningful role assignment

    confidence = _score_confidence(target, groups, explanatory, op, df)
    reasoning = _build_reasoning(target, groups, explanatory, op)

    return SemanticRoleAssignment(
        target_variable=target,
        grouping_variables=groups,
        explanatory_variables=explanatory,
        metric_variable=target,
        time_variable=time_var,
        operation=op,
        confidence=confidence,
        reasoning=reasoning,
    )


# ── Operation Classification ─────────────────────────────────────────────

_PREVALENCE_MARKERS = (
    "prevalence", "prevalent", "prevalences",
    "how common", "how frequent", "how widespread",
    "proportion of", "share of", "percentage of", "percent of",
    "rate of", "rates of", "incidence of", "incidence rate",
    "fraction of", "likelihood of",
    # Bare "rate" with outcome context ("churn rate", "default rate")
    "churn rate", "default rate", "attrition rate", "fraud rate",
    "failure rate", "dropout rate", "turnover rate", "mortality rate",
    "defect rate", "error rate", "incident rate",
)

_ASSOCIATION_MARKERS = (
    "associated with", "association with", "associations with",
    "risk factor", "risk factors", "contribute to", "contributes to",
    "predict", "predictors of", "predictor of", "predictive of",
    "linked to", "linked with", "related to risk",
    "most associated", "strongly associated", "appear associated",
    "combinations of risk", "combination of risk",
    "combinations of factor", "combination of factor",
    "what factors", "which factors", "what variables", "which variables",
    "what drives", "what influences", "what affects",
    "determinants of", "drivers of",
)

_COMPARISON_MARKERS = (
    "compare ", "comparison of", "difference between",
    "differ between", "differs between",
    "higher among", "lower among",
    "more common among", "less common among",
)

_TREND_MARKERS = (
    "over time", "trend in", "trends in", "trending",
    "become more common", "become less common", "becomes more common",
    "increase over", "decrease over", "change over",
    "more common among older", "more common among younger",
    "shift across age", "across age group",
)

_VISUALIZATION_MARKERS = (
    "visualization of", "visualize ", "visualise ",
    "build a chart of", "build chart of",
    "plot of", "build a plot",
    "build a visualization", "build a visualisation",
    "create a chart", "create a visualization",
    "create a plot", "show a chart",
)


def _classify_analysis_operation(normalized: str) -> str:
    """Classify the type of analysis the question is asking for."""
    # Check in priority order — prevalence + comparison can co-occur
    has_prevalence = any(m in normalized for m in _PREVALENCE_MARKERS)
    has_association = any(m in normalized for m in _ASSOCIATION_MARKERS)
    has_comparison = any(m in normalized for m in _COMPARISON_MARKERS)
    has_trend = any(m in normalized for m in _TREND_MARKERS)
    has_visualization = any(m in normalized for m in _VISUALIZATION_MARKERS)

    # "compare X prevalence between Y" → prevalence (with grouping)
    if has_prevalence:
        return "prevalence"
    # Detect "X rate by Y" pattern even when the specific rate type
    # isn't in the marker list — if the word before "rate" is an outcome concept
    if _has_outcome_rate_pattern(normalized):
        return "prevalence"
    if has_association:
        return "association"
    # "become more common among older" → trend with prevalence semantics
    if has_trend and _has_indicator_language(normalized):
        return "prevalence"
    if has_comparison and _has_indicator_language(normalized):
        return "prevalence"
    if has_visualization and _has_outcome_language(normalized):
        return "prevalence"
    # Plain comparison/trend/visualization without outcome language → not role-assignable
    return ""


def _has_outcome_rate_pattern(normalized: str) -> bool:
    """Detect 'X rate by Y' or 'X risk by Y' patterns for outcome concepts."""
    # Match patterns like "churn rate", "default risk", etc. even without
    # explicit prevalence markers — if an outcome-like word precedes 'rate'/'risk'
    outcome_words = (
        "churn", "default", "fraud", "attrition", "failure", "dropout",
        "turnover", "mortality", "defect", "error", "incident", "disease",
        "survival", "readmission", "complaint", "violation", "outcome",
    )
    for word in outcome_words:
        if f"{word} rate" in normalized or f"{word} risk" in normalized:
            return True
    return False


def _has_indicator_language(normalized: str) -> bool:
    """Check if the question references outcomes, indicators, or conditions."""
    markers = (
        "disease", "condition", "diagnosis", "indicator", "indicators",
        "health", "risk", "symptom", "symptoms", "outcome", "outcomes",
        "churn", "default", "fraud", "attrition", "failure", "dropout",
        "mortality", "morbidity", "readmission", "survival",
        "defect", "incident", "event",
    )
    return any(m in normalized for m in markers)


def _has_outcome_language(normalized: str) -> bool:
    """Check if visualization mentions an outcome/prevalence concept."""
    markers = (
        "prevalence", "rate", "incidence", "proportion",
        "disease", "churn", "default", "fraud", "attrition",
        "outcome", "failure", "dropout", "mortality",
    )
    return any(m in normalized for m in markers)


# ── Target/Outcome Variable Detection ────────────────────────────────────

# Generic outcome-like column name markers (no dataset-specific names)
_OUTCOME_NAME_MARKERS = (
    "disease", "attack", "churn", "default", "defaulted",
    "fraud", "fraudulent", "attrition", "survived", "survival",
    "passed", "failed", "failure", "approved", "rejected",
    "flag", "outcome", "event", "incident",
    "positive", "negative", "diagnosed", "diagnosis",
    "readmit", "readmission", "dropout", "turnover",
    "mortality", "morbidity", "defect", "defective",
    "complaint", "violation", "recidivism",
)


def _detect_target_variable(
    normalized_question: str,
    df: pd.DataFrame,
) -> str | None:
    """Detect the target/outcome variable generically.

    Uses structural heuristics (binary columns) and semantic name matching.
    """
    candidates: list[tuple[float, str]] = []

    for col in df.columns:
        name = str(col)
        series = df[col]
        score = 0.0

        is_binary = _is_binary_column(series)
        has_outcome_name = _has_outcome_name(name)

        if not is_binary and not has_outcome_name:
            continue

        # Base score for being a binary outcome
        if is_binary:
            score += 30
        if has_outcome_name:
            score += 40

        # Bonus: question mentions concepts from the column name
        name_tokens = set(_normalize(name).split())
        question_tokens = set(normalized_question.split())
        overlap = name_tokens & question_tokens
        if overlap:
            score += 25 * len(overlap)
        name_compact_tokens = _normalize(name).replace(" ", "")
        for token in question_tokens:
            if len(token) >= 4 and token in name_compact_tokens:
                score += 20

        # Bonus: column name contains a substring from the question's
        # key concept (e.g. "heart disease" → "heartdisease" in column)
        col_compact = _normalize(name).replace(" ", "")
        for concept in _extract_outcome_concepts(normalized_question):
            if concept in col_compact or col_compact in concept:
                score += 35

        # Penalty: identifier-like names
        if _is_identifier_name(name):
            score -= 100
        if any(marker in _normalize(name) for marker in ("cost", "fee", "payment", "admin", "access")) and not any(marker in normalized_question for marker in ("cost", "fee", "payment", "admin", "access")):
            score -= 35

        # Penalty: too many unique values for a binary flag
        nunique = int(series.nunique(dropna=True))
        if nunique > 10:
            score -= 20
        if nunique > 50:
            score -= 40

        if score > 0:
            candidates.append((score, name))

    if not candidates:
        return None

    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1]


def _is_binary_column(series: pd.Series) -> bool:
    """Check if a column contains only binary values (0/1, Yes/No, True/False)."""
    non_null = series.dropna()
    if non_null.empty:
        return False
    unique = set(non_null.unique())
    if len(unique) > 2:
        return False
    if len(unique) < 2:
        # Single-value columns can still be binary flags (e.g. all 1s)
        pass

    # Numeric binary
    if pd.api.types.is_numeric_dtype(series):
        numeric_unique = {float(v) for v in unique}
        if numeric_unique <= {0.0, 1.0}:
            return True

    # String binary
    str_unique = {str(v).lower().strip() for v in unique}
    binary_pairs = (
        {"yes", "no"}, {"true", "false"}, {"0", "1"},
        {"y", "n"}, {"t", "f"}, {"positive", "negative"},
        {"present", "absent"}, {"male", "female"},
    )
    # Male/Female is demographic, not outcome — exclude it
    if str_unique <= {"male", "female"} or str_unique <= {"m", "f"}:
        return False
    for pair in binary_pairs:
        if str_unique <= pair:
            return True

    return False


def _has_outcome_name(name: str) -> bool:
    """Check if a column name semantically suggests an outcome/target variable."""
    normalized = _normalize(name)
    return any(marker in normalized for marker in _OUTCOME_NAME_MARKERS)


def _extract_outcome_concepts(normalized_question: str) -> list[str]:
    """Extract likely outcome concepts from the question text.

    For "heart disease prevalence", extracts "heartdisease".
    For "churn risk", extracts "churn".
    """
    concepts: list[str] = []

    # Multi-word concepts before outcome markers
    outcome_triggers = (
        "prevalence", "rate", "incidence", "risk",
        "outcome", "status", "flag", "indicator",
    )
    for trigger in outcome_triggers:
        pattern = rf"(\w[\w\s]{{2,30}}?)\s+{trigger}"
        match = re.search(pattern, normalized_question)
        if match:
            concept = match.group(1).strip().replace(" ", "")
            if len(concept) >= 3:
                concepts.append(concept)

    # Concepts after "of" in prevalence context
    of_pattern = re.search(r"prevalence\s+of\s+(\w[\w\s]{2,30}?)(?:\s+(?:by|between|across|among|in|for)\b|$)", normalized_question)
    if of_pattern:
        concept = of_pattern.group(1).strip().replace(" ", "")
        if len(concept) >= 3:
            concepts.append(concept)

    # Concepts after "associated with"
    assoc_pattern = re.search(r"associated\s+with\s+(\w[\w\s]{2,30}?)(?:\s*[?.]|$)", normalized_question)
    if assoc_pattern:
        concept = assoc_pattern.group(1).strip().replace(" ", "")
        if len(concept) >= 3:
            concepts.append(concept)

    return concepts


def _is_identifier_name(name: str) -> bool:
    """Check if a column name looks like an identifier."""
    lowered = name.lower()
    if lowered == "id" or lowered.endswith("_id") or lowered.endswith(" id"):
        return True
    if "uuid" in lowered or "code" in lowered or "key" in lowered:
        return True
    return False


# ── Grouping Variable Detection ──────────────────────────────────────────

_GROUPING_PREPOSITIONS = (
    "between ", "by ", "across ", "per ", "among ",
    "for each ", "within ", "grouped by ", "split by ",
)


def _detect_grouping_variables(
    normalized_question: str,
    df: pd.DataFrame,
    *,
    exclude: set[str],
) -> list[str]:
    """Detect grouping variables from relational markers in the question."""
    groups: list[str] = []
    columns = [str(col) for col in df.columns if str(col) not in exclude]
    normalized_columns = {_normalize(str(col)): str(col) for col in columns}

    # 1. Direct column name mentions after grouping prepositions
    for prep in _GROUPING_PREPOSITIONS:
        idx = normalized_question.find(prep)
        if idx < 0:
            continue
        after = normalized_question[idx + len(prep):]
        # Try to match column names in the text after the preposition
        for norm_col, original_col in sorted(
            normalized_columns.items(), key=lambda x: -len(x[0])
        ):
            if norm_col in after and original_col not in groups:
                groups.append(original_col)

    # 2. "between X and Y" pattern for binary grouping
    between_match = re.search(
        r"between\s+(\w[\w\s]*?)\s+and\s+(\w[\w\s]*?)(?:\s*[?.!,]|$)",
        normalized_question,
    )
    if between_match:
        for group_text in (between_match.group(1), between_match.group(2)):
            group_text = group_text.strip()
            # Try to resolve to a column whose values contain these terms
            resolved = _resolve_grouping_value_to_column(group_text, df, exclude=exclude)
            if resolved and resolved not in groups:
                groups.append(resolved)

    # 3. Columns mentioned in the question that are categorical/binary
    # (but not already in the groups list and not the target)
    if not groups:
        for norm_col, original_col in normalized_columns.items():
            if norm_col in normalized_question and original_col not in groups:
                series = df[original_col]
                # Only pick categoricals or binary columns as grouping
                if _is_binary_column(series) or _is_categorical_grouping(series):
                    groups.append(original_col)

    return groups[:3]  # Limit to 3 grouping variables


def _resolve_grouping_value_to_column(
    value_text: str,
    df: pd.DataFrame,
    *,
    exclude: set[str],
) -> str | None:
    """Resolve a value mention (e.g. 'smokers') to the column that contains it."""
    normalized_value = _normalize(value_text)
    for col in df.columns:
        name = str(col)
        if name in exclude:
            continue
        # Check if value text matches the column name
        if normalized_value in _normalize(name) or _normalize(name) in normalized_value:
            return name
        # Check if column values contain this text
        series = df[col]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue
        unique_values = {_normalize(str(v)) for v in series.dropna().unique()[:50]}
        for uv in unique_values:
            if normalized_value in uv or uv in normalized_value:
                return name
    return None


def _is_categorical_grouping(series: pd.Series) -> bool:
    """Check if a series is suitable as a grouping variable."""
    if pd.api.types.is_numeric_dtype(series):
        return False
    unique = int(series.nunique(dropna=True))
    return 2 <= unique <= 30


# ── Explanatory Variable Detection ───────────────────────────────────────

def _detect_explanatory_variables(
    normalized_question: str,
    df: pd.DataFrame,
    *,
    target: str,
    groups: list[str],
) -> list[str]:
    """Detect explanatory/risk-factor variables for association analysis."""
    exclude = {target} | set(groups)
    explanatory: list[str] = []

    for col in df.columns:
        name = str(col)
        if name in exclude:
            continue
        if _is_identifier_name(name):
            continue
        series = df[col]
        # Include binary columns, low-cardinality categoricals, and numeric columns
        if _is_binary_column(series):
            explanatory.append(name)
        elif _is_categorical_grouping(series):
            explanatory.append(name)
        elif pd.api.types.is_numeric_dtype(series):
            # Exclude identifier-like numeric columns
            non_null = series.dropna()
            if non_null.empty:
                continue
            unique_ratio = non_null.nunique() / max(len(non_null), 1)
            if unique_ratio < 0.9:  # Not identifier-like
                explanatory.append(name)

    return explanatory[:15]  # Limit to top 15


# ── Time Variable Detection ──────────────────────────────────────────────

def _detect_time_variable(
    normalized_question: str,
    df: pd.DataFrame,
) -> str | None:
    """Detect time/temporal variable from the dataset."""
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return str(col)
    time_markers = ("date", "time", "year", "month", "timestamp", "period")
    for col in df.columns:
        name = _normalize(str(col))
        if any(m in name for m in time_markers):
            return str(col)
    return None


# ── Scoring & Reasoning ──────────────────────────────────────────────────

def _score_confidence(
    target: str | None,
    groups: list[str],
    explanatory: list[str],
    operation: str,
    df: pd.DataFrame,
) -> float:
    """Score confidence of the role assignment."""
    score = 0.0
    if target:
        score += 0.4
        if _is_binary_column(df[target]):
            score += 0.2  # Binary targets are more certain
    if groups:
        score += 0.2
    if operation in ("prevalence", "association"):
        score += 0.1
    if explanatory:
        score += 0.1
    return min(score, 1.0)


def _build_reasoning(
    target: str | None,
    groups: list[str],
    explanatory: list[str],
    operation: str,
) -> str:
    """Build human-readable reasoning for the role assignment."""
    parts: list[str] = []
    parts.append(f"Operation: {operation}")
    if target:
        parts.append(f"Target/outcome: {target}")
    if groups:
        parts.append(f"Grouping: {', '.join(groups)}")
    if explanatory:
        parts.append(f"Explanatory: {', '.join(explanatory[:5])}")
    return "; ".join(parts)


# ── Analysis Response Builders ────────────────────────────────────────────

def prevalence_analysis_response(
    question: str,
    df: pd.DataFrame,
    roles: SemanticRoleAssignment,
) -> dict[str, Any] | None:
    """Build a prevalence/rate analysis response using the assigned roles.

    For binary target variables: mean(target) grouped by each grouping variable = prevalence.
    """
    target = roles.target_variable
    if not target or target not in df.columns:
        return None

    series = df[target]
    # Convert to numeric for mean computation
    numeric_target = pd.to_numeric(series, errors="coerce")
    if numeric_target.isna().all():
        # Try Yes/No → 1/0 conversion
        str_values = series.astype(str).str.lower().str.strip()
        mapping = {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0,
                   "positive": 1, "negative": 0, "present": 1, "absent": 0}
        numeric_target = str_values.map(mapping)
        if numeric_target.isna().all():
            return None

    # Overall prevalence
    overall = float(numeric_target.mean())

    if not roles.grouping_variables:
        # No grouping — just report overall prevalence
        summary = (
            f"The overall prevalence of `{target}` is {overall:.1%} "
            f"({int(numeric_target.sum())} out of {int(numeric_target.count())} records)."
        )
        return _role_output(
            question=question,
            summary=summary,
            findings=[summary],
            evidence=[f"Computed mean(`{target}`) across {int(numeric_target.count())} non-null rows."],
            target=target,
            operation="prevalence",
        )

    # Grouped prevalence
    all_rows: list[dict[str, Any]] = []
    all_findings: list[str] = []
    chart_rows: list[dict[str, Any]] = []
    group_candidates = [group for group in roles.grouping_variables if group in df.columns and group != target]
    ordered_groups = [group for group in group_candidates if _is_ordered_grouping_candidate(df[group], group)]
    primary_group = ordered_groups[0] if ordered_groups else group_candidates[0]
    secondary_group = next((group for group in group_candidates if group != primary_group), None)

    if primary_group not in df.columns:
        return None

    working = pd.DataFrame({primary_group: _analysis_group_series(df[primary_group], primary_group)})
    groupby_cols = [primary_group]
    if secondary_group:
        working[secondary_group] = _analysis_group_series(df[secondary_group], secondary_group)
        groupby_cols.append(secondary_group)
    working["_target"] = numeric_target.values

    grouped = (
        working.groupby(groupby_cols, dropna=False)["_target"]
        .agg(["mean", "count", "sum"])
        .reset_index()
        .rename(columns={"mean": "prevalence", "count": "total", "sum": "positive_count"})
    )
    grouped["prevalence"] = grouped["prevalence"].round(4)
    ordered_primary = _is_ordered_grouping_candidate(df[primary_group], primary_group)
    if ordered_primary:
        grouped["_sort"] = grouped[primary_group].map(_group_sort_key)
        grouped = grouped.sort_values(["_sort", secondary_group] if secondary_group else ["_sort"])
    else:
        grouped = grouped.sort_values("prevalence", ascending=False)

    for _, row in grouped.iterrows():
        group_val = str(row[primary_group])
        secondary_val = str(row[secondary_group]) if secondary_group else ""
        prev = float(row["prevalence"])
        total = int(row["total"])
        pos = int(row["positive_count"])
        result_row = {
            "group": group_val,
            "prevalence": round(prev, 4),
            "prevalence_pct": f"{prev:.1%}",
            "positive_count": pos,
            "total": total,
        }
        if secondary_group:
            result_row["secondary_group"] = secondary_val
            result_row["group_label"] = f"{group_val} / {secondary_val}"
        all_rows.append(result_row)
        chart_point = {"x": group_val, "y": round(prev * 100, 2)}
        if secondary_group:
            chart_point["series"] = secondary_val
        chart_rows.append(chart_point)

    if len(all_rows) >= 2:
        if ordered_primary:
            low = all_rows[0]
            high = all_rows[-1]
            diff = float(high["prevalence"]) - float(low["prevalence"])
            direction = "increases" if diff > 0 else "decreases" if diff < 0 else "is flat"
            summary = (
                f"`{target}` prevalence by ordered `{primary_group}` {direction} from "
                f"`{low.get('group_label', low['group'])}` ({low['prevalence_pct']}) to "
                f"`{high.get('group_label', high['group'])}` ({high['prevalence_pct']}), "
                f"a change of {diff:.1%}. Overall prevalence is {overall:.1%} across {int(numeric_target.count())} records."
            )
        else:
            top = all_rows[0]
            bottom = all_rows[-1]
            diff = float(top["prevalence"]) - float(bottom["prevalence"])
            summary = (
                f"`{target}` prevalence is highest among {primary_group} = `{top.get('group_label', top['group'])}` "
                f"({top['prevalence_pct']}) and lowest among `{bottom['group']}` ({bottom['prevalence_pct']}), "
                f"a gap of {diff:.1%}. "
                f"Overall prevalence is {overall:.1%} across {int(numeric_target.count())} records."
            )
        all_findings.append(summary)
        all_findings.append(
            f"The difference between compared groups ({abs(diff):.1%}) "
            f"suggests that `{primary_group}` may be associated with `{target}`, "
            f"but this is an association, not a causal claim."
        )
    else:
        summary = (
            f"`{target}` prevalence grouped by `{primary_group}`: "
            + ", ".join(f"{r['group']}={r['prevalence_pct']}" for r in all_rows)
            + f". Overall: {overall:.1%}."
        )
        all_findings.append(summary)

    return _role_output(
        question=question,
        summary=summary,
        findings=all_findings,
        evidence=[
            f"Computed mean(`{target}`) grouped by `{primary_group}` across {int(numeric_target.count())} rows.",
        ],
        target=target,
        operation="prevalence",
        result_rows=all_rows,
        chart_rows=chart_rows,
        chart_title=f"{target} prevalence by {primary_group}" + (f" and {secondary_group}" if secondary_group else ""),
        chart_x=primary_group,
        chart_y=f"{target} prevalence (%)",
        code=f"df.groupby({groupby_cols!r})[{target!r}].mean()",
    )


def _is_ordered_grouping_candidate(series: pd.Series, name: str) -> bool:
    normalized = _normalize(name)
    if any(marker in normalized for marker in ("age", "tenure", "grade", "level", "severity", "income", "elevation", "band", "bracket")):
        return True
    return pd.api.types.is_numeric_dtype(series) and int(series.nunique(dropna=True)) > 5


def _analysis_group_series(series: pd.Series, name: str) -> pd.Series:
    if not _is_ordered_grouping_candidate(series, name) or not pd.api.types.is_numeric_dtype(series):
        return series.astype("object")
    numeric = pd.to_numeric(series, errors="coerce")
    unique = int(numeric.nunique(dropna=True))
    if unique <= 6:
        return numeric.astype("object")
    ranked = numeric.rank(method="first")
    try:
        binned = pd.qcut(ranked, q=min(5, unique), duplicates="drop")
        labels = binned.map(lambda interval: f"{numeric[binned == interval].min():g}-{numeric[binned == interval].max():g}" if pd.notna(interval) else "Missing")
        return labels.astype("object")
    except Exception:
        return numeric.astype("object")


def _group_sort_key(value: Any) -> tuple[float, str]:
    text = str(value)
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        try:
            return float(numbers[0]), text
        except ValueError:
            pass
    return 0.0, text


def association_analysis_response(
    question: str,
    df: pd.DataFrame,
    roles: SemanticRoleAssignment,
) -> dict[str, Any] | None:
    """Build an association/risk-factor analysis response.

    For each explanatory variable, compute prevalence of target
    within each category and rank by prevalence difference.
    """
    target = roles.target_variable
    if not target or target not in df.columns:
        return None
    if not roles.explanatory_variables:
        return None

    series = df[target]
    numeric_target = pd.to_numeric(series, errors="coerce")
    if numeric_target.isna().all():
        str_values = series.astype(str).str.lower().str.strip()
        mapping = {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0,
                   "positive": 1, "negative": 0, "present": 1, "absent": 0}
        numeric_target = str_values.map(mapping)
        if numeric_target.isna().all():
            return None

    overall = float(numeric_target.mean())
    associations: list[dict[str, Any]] = []

    for var in roles.explanatory_variables[:10]:
        if var not in df.columns:
            continue
        col_series = df[var]

        if _is_binary_column(col_series) or (
            pd.api.types.is_object_dtype(col_series) and int(col_series.nunique(dropna=True)) <= 15
        ):
            # Categorical: compute prevalence per group
            working = pd.DataFrame({var: col_series, "_target": numeric_target.values})
            grouped = working.groupby(var, dropna=False)["_target"].agg(["mean", "count"]).reset_index()
            if grouped.empty or len(grouped) < 2:
                continue
            max_prev = float(grouped["mean"].max())
            min_prev = float(grouped["mean"].min())
            diff = max_prev - min_prev
            best_group = str(grouped.loc[grouped["mean"].idxmax(), var])
            associations.append({
                "variable": var,
                "type": "categorical",
                "max_prevalence": round(max_prev, 4),
                "min_prevalence": round(min_prev, 4),
                "prevalence_gap": round(diff, 4),
                "prevalence_gap_pct": f"{diff:.1%}",
                "strongest_group": best_group,
                "groups": int(len(grouped)),
            })
        elif pd.api.types.is_numeric_dtype(col_series):
            # Numeric: compute correlation with target
            corr = float(numeric_target.corr(pd.to_numeric(col_series, errors="coerce")))
            if pd.isna(corr):
                continue
            associations.append({
                "variable": var,
                "type": "numeric",
                "correlation": round(corr, 4),
                "abs_correlation": round(abs(corr), 4),
                "prevalence_gap": round(abs(corr), 4),  # use abs_corr for ranking
                "prevalence_gap_pct": f"{abs(corr):.2f}",
            })

    if not associations:
        return None

    # Sort by prevalence gap (or abs correlation)
    associations.sort(key=lambda x: -x["prevalence_gap"])

    top = associations[0]
    summary_parts = [
        f"`{top['variable']}` shows the strongest association with `{target}` "
        f"(prevalence gap: {top['prevalence_gap_pct']})."
    ]
    if len(associations) > 1:
        second = associations[1]
        summary_parts.append(
            f"`{second['variable']}` is second (gap: {second['prevalence_gap_pct']})."
        )
    summary_parts.append(f"Overall `{target}` prevalence is {overall:.1%}.")
    summary = " ".join(summary_parts)

    findings = [summary]
    findings.append(
        "These are statistical associations, not causal claims. "
        "Confounding variables may explain some observed gaps."
    )

    result_rows = associations[:10]
    chart_rows = [
        {"x": a["variable"], "y": round(a["prevalence_gap"] * 100, 2)}
        for a in associations[:10]
    ]

    return _role_output(
        question=question,
        summary=summary,
        findings=findings,
        evidence=[
            f"Analyzed {len(associations)} variables against `{target}` "
            f"across {int(numeric_target.count())} records.",
        ],
        target=target,
        operation="association",
        result_rows=result_rows,
        chart_rows=chart_rows,
        chart_title=f"Association strength with {target}",
        chart_x="Variable",
        chart_y="Association strength (%)",
        code=f"# Prevalence gap analysis against {target!r}",
    )


# ── Output Builder ────────────────────────────────────────────────────────

def _role_output(
    *,
    question: str,
    summary: str,
    findings: list[str],
    evidence: list[str],
    target: str,
    operation: str,
    result_rows: list[dict[str, Any]] | None = None,
    chart_rows: list[dict[str, Any]] | None = None,
    chart_title: str = "",
    chart_x: str = "",
    chart_y: str = "",
    code: str = "",
) -> dict[str, Any]:
    """Build a standard output dict for role-based analysis."""
    artifacts: list[dict[str, Any]] = []
    if result_rows:
        artifacts.append({
            "artifact_type": "table",
            "title": chart_title or f"{target} analysis",
            "content": result_rows,
            "visibility": "user",
            "pinned": True,
            "metadata": {"target": target, "analysis_type": operation},
        })
    if chart_rows:
        artifacts.append({
            "artifact_type": "chart",
            "title": chart_title or f"{target} analysis",
            "content": {
                "chart_type": "bar",
                "x": "x",
                "y": "y",
                "metric": chart_y or target,
                "rows": chart_rows,
            },
            "visibility": "user",
            "pinned": True,
            "metadata": {"target": target, "analysis_type": operation},
        })

    return {
        "summary": summary,
        "final_answer": summary,
        "structured_report": {
            "question": question,
            "summary": summary,
            "key_findings": findings[:3],
            "evidence": evidence,
            "limitations": [
                "Association does not imply causation. Confounding variables may explain observed differences.",
            ],
            "next_steps": [
                f"Break down `{target}` by additional grouping variables to check if the pattern holds.",
                "Look for confounding variables that may co-vary with the grouping dimension.",
            ],
        },
        "key_findings": findings[:3],
        "limitations": [
            "Association does not imply causation.",
        ],
        "artifacts": artifacts,
        "tool_timeline": [{"tool": f"semantic_role_{operation}", "status": "ok"}],
        "trace_metadata": {
            "fallback": f"semantic_role_{operation}",
            "analysis_type": f"semantic_role_{operation}",
            "target_variable": target,
            "operation": operation,
        },
        "generated_code": code,
    }


# ── Utilities ─────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    return " ".join(str(text or "").replace("_", " ").replace("-", " ").casefold().split())
