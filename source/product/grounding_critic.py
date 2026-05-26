"""Check that synthesis text is grounded in computed evidence."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from source.product.plan_executor import EvidencePackage
from source.product.grounded_synthesis import SynthesisResult


@dataclass
class CriticResult:
    """Stores grounding issues and an optional cleaned answer."""

    passed: bool
    has_critical_failures: bool = False
    issues: list[str] = field(default_factory=list)
    cleaned_synthesis: SynthesisResult | None = None


def validate_synthesis(
    synthesis: SynthesisResult,
    evidence: EvidencePackage,
    *,
    question_domain: str | None = None,
    dataset_domain: str | None = None,
) -> CriticResult:
    """Validate that an answer only uses computed evidence."""
    issues: list[str] = []

    answer_numbers = _extract_numbers(synthesis.answer)
    evidence_numbers = _collect_evidence_numbers(evidence)
    ungrounded_numbers = [n for n in answer_numbers if not _number_in_evidence(n, evidence_numbers)]
    if ungrounded_numbers:
        issues.append(f"Ungrounded numbers in answer: {ungrounded_numbers[:5]}")

    ungrounded_columns = _check_column_references(synthesis.answer, evidence)
    if ungrounded_columns:
        issues.append(f"References to unknown columns: {ungrounded_columns}")

    ungrounded_findings = []
    for i, finding in enumerate(synthesis.findings):
        finding_numbers = _extract_numbers(finding.get("evidence", ""))
        bad_nums = [n for n in finding_numbers if not _number_in_evidence(n, evidence_numbers)]
        if bad_nums:
            ungrounded_findings.append(i)
            issues.append(f"Finding #{i + 1} has ungrounded numbers: {bad_nums[:3]}")

    hallucination_markers = _check_hallucination_markers(synthesis.answer)
    if hallucination_markers:
        issues.append(f"Possible hallucination: {hallucination_markers}")

    if question_domain and dataset_domain and question_domain != "general":
        domain_issues = _check_domain_relevance(
            synthesis.answer, question_domain, dataset_domain,
        )
        if domain_issues:
            issues.append(f"Domain relevance: {domain_issues}")

    has_critical = len(ungrounded_numbers) > 3 or len(hallucination_markers) > 0
    passed = len(issues) == 0

    result = CriticResult(
        passed=passed,
        has_critical_failures=has_critical,
        issues=issues,
    )

    if issues and not has_critical:
        cleaned_findings = [
            f for i, f in enumerate(synthesis.findings)
            if i not in set(ungrounded_findings)
        ]
        result.cleaned_synthesis = SynthesisResult(
            answer=synthesis.answer,
            findings=cleaned_findings if cleaned_findings else synthesis.findings,
            limitations=synthesis.limitations + [f"Critic note: {len(issues)} grounding issue(s) detected"],
            next_steps=synthesis.next_steps,
        )

    return result


def _extract_numbers(text: str) -> list[float]:
    """Extract numbers from text."""
    matches = re.findall(r"-?\d+(?:\.\d+)?", str(text or ""))
    result: list[float] = []
    for m in matches:
        try:
            result.append(float(m))
        except ValueError:
            continue
    return result


def _collect_evidence_numbers(evidence: EvidencePackage) -> set[float]:
    """Collect all numbers from evidence tables and statistics."""
    numbers: set[float] = set()

    # From computed tables
    for table in evidence.computed_tables:
        for row in table.get("rows", []):
            if isinstance(row, dict):
                for value in row.values():
                    try:
                        numbers.add(float(value))
                    except (TypeError, ValueError):
                        continue
        # From other table fields
        for key in ("before_count", "after_count", "threshold"):
            val = table.get(key)
            if val is not None:
                try:
                    numbers.add(float(val))
                except (TypeError, ValueError):
                    pass
        # From shifts
        for shift in table.get("shifts", []):
            if isinstance(shift, dict):
                for v in shift.values():
                    try:
                        numbers.add(float(v))
                    except (TypeError, ValueError):
                        pass

    # From summary statistics
    _collect_dict_numbers(evidence.summary_statistics, numbers)

    # Meta numbers
    numbers.add(float(evidence.record_count))

    return numbers


def _collect_dict_numbers(data: Any, numbers: set[float]) -> None:
    """Recursively collect numbers from a nested dict."""
    if isinstance(data, dict):
        for value in data.values():
            _collect_dict_numbers(value, numbers)
    elif isinstance(data, (list, tuple)):
        for item in data:
            _collect_dict_numbers(item, numbers)
    else:
        try:
            numbers.add(float(data))
        except (TypeError, ValueError):
            pass


def _number_in_evidence(number: float, evidence_numbers: set[float]) -> bool:
    """Check if a number matches any evidence number (with tolerance)."""
    if number in evidence_numbers:
        return True
    # Allow rounding tolerance
    for en in evidence_numbers:
        if abs(number - en) < 0.15:
            return True
        # Allow percentage derivations (number could be share computed from evidence)
        if en != 0 and abs(number - en * 100) < 0.5:
            return True
    # Small integers (1-10) are common in language and don't need grounding
    if abs(number) <= 10 and number == int(number):
        return True
    return False


# ── Column reference checking ───────────────────────────────────────────────

def _check_column_references(text: str, evidence: EvidencePackage) -> list[str]:
    """Check for references to columns not in evidence."""
    # We only check if text contains backtick-quoted names not in evidence
    quoted = re.findall(r"`([^`]+)`", text)
    if not quoted:
        return []
    known_cols = set(evidence.columns_used)
    # Also add table names and field names from evidence
    for table in evidence.computed_tables:
        for key in ("dimension", "metric", "category_field", "group_column", "entity_column", "time_field", "duration_field"):
            val = table.get(key)
            if val:
                known_cols.add(val)
    for key in evidence.summary_statistics:
        known_cols.add(key)
    unknown = [q for q in quoted if q not in known_cols and q.lower() not in {c.lower() for c in known_cols}]
    return unknown


# ── Hallucination detection ─────────────────────────────────────────────────

_HALLUCINATION_MARKERS = (
    "as we can see from the raw data",
    "based on my domain knowledge",
    "typically in this industry",
    "research shows that",
    "studies indicate",
    "it is well known that",
    "historically speaking",
    "according to best practices",
)


def _check_hallucination_markers(text: str) -> list[str]:
    """Check for language suggesting the LLM is using knowledge beyond the evidence."""
    normalized = text.lower()
    return [marker for marker in _HALLUCINATION_MARKERS if marker in normalized]


# ── Domain relevance checking ──────────────────────────────────────────────

_DOMAIN_INDICATOR_TERMS: dict[str, tuple[str, ...]] = {
    "healthcare": ("diagnosis", "patient", "treatment", "clinical", "disease", "symptom", "medical", "hospital"),
    "entertainment": ("movie", "tv show", "genre", "director", "cast", "rating", "film", "streaming"),
    "retail": ("sales", "revenue", "profit", "customer", "product", "order", "discount", "shipping"),
    "financial": ("income", "salary", "loan", "credit", "investment", "portfolio", "interest", "mortgage"),
    "scientific": ("species", "habitat", "organism", "ecosystem", "experiment", "gene"),
    "education": ("student", "grade", "course", "teacher", "school", "university"),
    "hr": ("employee", "hire", "attrition", "department", "performance", "tenure"),
}


def _check_domain_relevance(
    answer: str,
    question_domain: str,
    dataset_domain: str,
) -> str:
    """Check if the answer contains entities from the wrong domain.

    Returns a description of the mismatch, or empty string if OK.
    """
    if question_domain == dataset_domain:
        return ""

    normalized = answer.lower()

    # Check if answer uses dataset domain terms when question is from different domain
    dataset_terms = _DOMAIN_INDICATOR_TERMS.get(dataset_domain, ())
    question_terms = _DOMAIN_INDICATOR_TERMS.get(question_domain, ())

    dataset_hits = [t for t in dataset_terms if t in normalized]
    question_hits = [t for t in question_terms if t in normalized]

    # If the answer uses many dataset-domain terms but no question-domain terms,
    # it's answering with wrong-domain content
    if len(dataset_hits) >= 2 and not question_hits:
        return (
            f"Answer uses {dataset_domain} terms ({', '.join(dataset_hits[:3])}) "
            f"but question is about {question_domain}"
        )

    return ""


# ── Critic backstop for final answer validation ─────────────────────────────

def critic_backstop_check(
    question: str,
    final_answer: str,
    df: Any = None,
) -> str | None:
    """Last-resort backstop: catch any remaining compatibility bypass.

    If the final answer mentions wrong-domain columns or entities while
    the question domain is incompatible, return the compatibility refusal.
    Returns None if the answer passes the check.
    """
    if not question or not final_answer or df is None:
        return None

    try:
        from source.product.compatibility_engine import (
            enforce_question_dataset_compatibility,
            check_question_dataset_compatibility,
        )
        import pandas as pd

        if not isinstance(df, pd.DataFrame) or df.empty:
            return None

        decision = enforce_question_dataset_compatibility(question, df)
        if decision.allowed:
            return None

        # Question is incompatible — check if the answer mentions dataset columns
        compat = check_question_dataset_compatibility(question, df)
        if compat is None:
            return None

        # The answer is from an incompatible question — return the refusal
        return compat.reason
    except Exception:
        return None


def dataframe_operation_precedence_check(question: str, final_answer: str) -> str | None:
    """Reject dtype-inspection answers when the user asked for analysis.

    Mentions of "column" or "numeric metric" can appear in perfectly valid
    analytical requests. Schema/type inspection is only acceptable when that is
    the actual user intent.
    """
    question_lower = str(question or "").casefold()
    answer_lower = str(final_answer or "").casefold()
    if not question_lower or not answer_lower:
        return None

    analytical_markers = (
        "top ",
        "highest",
        "lowest",
        "rank",
        "ranking",
        "group by",
        "grouping column",
        " by total ",
        "total ",
        "average ",
        "sum ",
        "metric",
        "compare",
        "chart",
        "visualize",
        "visualise",
        "plot",
        "distribution",
    )
    dtype_answer_markers = (
        "dtype inspection",
        "dataframe dtype inspection",
        "checked dtypes",
        "column type",
        "column types",
        "data types",
    )
    if any(marker in question_lower for marker in analytical_markers) and any(
        marker in answer_lower for marker in dtype_answer_markers
    ):
        return "Analytical aggregation/ranking/chart intent must route to the analytical planner, not dtype inspection."
    return None


def explicit_constraint_plan_check(question: str, output: Any, df: Any = None) -> str | None:
    """Reject answers that lose locked user constraints before execution."""
    if not question or not isinstance(output, dict) or df is None:
        return None
    try:
        import pandas as pd
        from source.product.llm_semantic_planner import plan_explicit_constraints

        if not isinstance(df, pd.DataFrame) or df.empty:
            return None
        expected = plan_explicit_constraints(question, df)
        if not expected or not expected.constraints_locked:
            return None
        trace = output.get("trace_metadata") if isinstance(output.get("trace_metadata"), dict) else {}
        actual = trace.get("query_plan") if isinstance(trace.get("query_plan"), dict) else {}
        if not actual:
            for artifact in output.get("artifacts", []) if isinstance(output.get("artifacts"), list) else []:
                metadata = artifact.get("metadata") if isinstance(artifact, dict) and isinstance(artifact.get("metadata"), dict) else {}
                actual = metadata.get("query_plan") if isinstance(metadata.get("query_plan"), dict) else {}
                if actual:
                    break
        if not actual:
            return "Explicit analytical constraints were detected, but no validated query plan was attached to the answer."
        if expected.constraints_locked.get("metric_columns"):
            actual_metric = actual.get("metric") or (actual.get("metric_columns") or [None])[0]
            if actual_metric != expected.metric:
                return f"Explicit metric `{expected.metric}` was ignored or replaced by `{actual_metric}`."
        if expected.constraints_locked.get("grouping_columns"):
            actual_groups = actual.get("grouping_columns") or ([actual.get("dimension")] if actual.get("dimension") else [])
            missing_groups = [group for group in expected.grouping_columns if group not in actual_groups]
            if missing_groups:
                return f"Explicit grouping column(s) were ignored: {', '.join(missing_groups)}."
        if expected.constraints_locked.get("aggregation"):
            if actual.get("aggregation") != expected.aggregation:
                return f"Explicit aggregation `{expected.aggregation}` was ignored or replaced by `{actual.get('aggregation')}`."
            if actual.get("aggregation") in expected.forbidden_aggregations:
                return f"Aggregation `{actual.get('aggregation')}` was explicitly forbidden by the user."
        if expected.artifact_required:
            artifacts = output.get("artifacts") if isinstance(output.get("artifacts"), list) else []
            if not any((artifact.get("artifact_type") or artifact.get("type")) == "chart" for artifact in artifacts if isinstance(artifact, dict)):
                return "Visualization was explicitly requested, but no chart artifact was produced."
            expected_chart_type = _explicit_chart_type(question)
            chart_artifacts = [artifact for artifact in artifacts if isinstance(artifact, dict) and (artifact.get("artifact_type") or artifact.get("type")) == "chart"]
            if expected_chart_type and chart_artifacts:
                chart = chart_artifacts[0]
                content = chart.get("content") if isinstance(chart.get("content"), dict) else {}
                metadata = chart.get("metadata") if isinstance(chart.get("metadata"), dict) else {}
                actual_chart_type = str(content.get("chart_type") or metadata.get("chart_type") or "").casefold()
                if expected_chart_type not in actual_chart_type:
                    return f"Explicit chart type `{expected_chart_type}` was silently changed to `{actual_chart_type or 'unknown'}`."
                if expected_chart_type == "scatter" and not _scatter_artifact_has_xy(content):
                    return "Scatter/relationship chart artifact is missing aligned x/y values."
        return None
    except Exception:
        return None


def _explicit_chart_type(question: str) -> str | None:
    text = str(question or "").casefold()
    if any(marker in text for marker in ("scatter plot", "scatter chart", "relationship chart", " versus ", " vs ")):
        return "scatter"
    if any(marker in text for marker in ("line chart", "trend chart", " over time", " trend over ")):
        return "line"
    if "bar chart" in text:
        return "bar"
    if "heatmap" in text:
        return "heatmap"
    if "histogram" in text:
        return "histogram"
    return None


def _scatter_artifact_has_xy(content: dict[str, Any]) -> bool:
    points = content.get("points")
    if isinstance(points, list):
        valid = [point for point in points if isinstance(point, dict) and _is_number(point.get("x")) and _is_number(point.get("y"))]
        return bool(valid) and len(valid) == len(points)
    rows = content.get("rows")
    x_key = content.get("x")
    y_key = content.get("y")
    if not isinstance(rows, list) or not isinstance(x_key, str) or not isinstance(y_key, str):
        return False
    valid = [row for row in rows if isinstance(row, dict) and _is_number(row.get(x_key)) and _is_number(row.get(y_key))]
    return bool(valid) and len(valid) == len(rows)


def _is_number(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def role_confusion_check(
    question: str,
    final_answer: str,
    df: Any = None,
) -> str | None:
    """Detect role confusion: grouping variable averaged as metric.

    Example: mean(grouping_flag) when the question asks about outcome prevalence.
    Returns a corrected response or None if the answer is fine.
    """
    if not question or not final_answer or df is None:
        return None

    try:
        import pandas as pd
        if not isinstance(df, pd.DataFrame) or df.empty:
            return None

        from source.product.semantic_role_assignment import assign_semantic_roles
        roles = assign_semantic_roles(question, df)
        if not roles or not roles.target_variable or roles.confidence < 0.5:
            return None

        # Check if the answer mentions averaging/mean of a grouping variable
        # instead of the target variable
        answer_lower = final_answer.lower()
        target_lower = roles.target_variable.lower().replace("_", " ")

        for group_var in roles.grouping_variables:
            group_lower = group_var.lower().replace("_", " ")
            # Detect patterns like "mean(GroupVar)" or "average of GroupVar"
            confusion_patterns = [
                f"mean({group_lower}",
                f"average({group_lower}",
                f"average of {group_lower}",
                f"mean of {group_lower}",
                f"`{group_lower}` has an average",
                f"`{group_lower}` has a mean",
                f"{group_lower} has an average",
                f"{group_lower} has a mean",
            ]
            if any(p in answer_lower for p in confusion_patterns):
                return (
                    f"The analysis incorrectly computed the average of `{group_var}` "
                    f"(a grouping variable), when the question asks about "
                    f"`{roles.target_variable}` (the outcome variable). "
                    f"The correct analysis is: prevalence of `{roles.target_variable}` "
                    f"grouped by `{group_var}`."
                )

        return None
    except Exception:
        return None


def stale_domain_leakage_check(
    final_answer: str,
    df: Any = None,
) -> list[str]:
    """Detect if the answer mentions domain-specific terms not present in the dataset.

    Catches stale semantic contamination where findings from a previous question
    (e.g., about movies/actors) leak into a response about a different domain
    (e.g., healthcare).
    """
    if not final_answer or df is None:
        return []
    try:
        import pandas as pd
        if not isinstance(df, pd.DataFrame) or df.empty:
            return []

        col_names_lower = {str(col).lower().replace("_", " ") for col in df.columns}
        all_col_text = " ".join(col_names_lower)
        answer_lower = final_answer.lower()

        # Check each domain's indicator terms against the dataset
        leaks: list[str] = []
        for domain, terms in _DOMAIN_INDICATOR_TERMS.items():
            for term in terms:
                # Term appears in answer but NOT in any column name or column values
                if term in answer_lower and term not in all_col_text:
                    leaks.append(f"'{term}' ({domain} domain) mentioned in answer but not in dataset columns")
        return leaks[:5]
    except Exception:
        return []
