from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

from source.llm.factory import make_llm


@dataclass(frozen=True)
class ReasoningIntent:
    kind: str
    mode: str
    requires_llm: bool


INTERPRETIVE_MODES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("customer_behavior_analysis", ("customer behavior", "buyer", "retention", "whale", "customers", "клиентск", "поведен")),
    ("risk_analysis", ("risk", "risks", "concerning", "concern", "dangerous", "опас", "риск", "тревож")),
    ("strategic_recommendation", ("recommend", "action", "management", "strategy", "should we", "product manager", "investigate first", "what would you investigate", "что делать", "рекоменд", "менеджмент")),
    ("executive_summary", ("story", "tell", "insight", "insights", "summary", "most important", "top priorities", "вывод", "инсайт", "история", "самые важные")),
    ("hypothesis_generation", ("hypothesis", "hypotheses", "why might", "possible cause", "гипотез", "причин")),
    ("operational_analysis", ("operational", "process", "delivery", "shipping", "операцион", "процесс", "достав")),
    ("revenue_quality_analysis", ("revenue quality", "concentration", "dependency", "volatility", "выручк", "концентрац", "волатиль")),
)

HYBRID_MARKERS = (
    "explain this chart",
    "explain the chart",
    "interpret chart",
    "interpret this",
    "compare distributions",
    "interpret outliers",
    "regional differences",
    "explain the adjusted chart",
    "объясни график",
    "объясни диаграм",
    "интерпрет",
)

COMPUTATIONAL_MARKERS = (
    "top ",
    "highest",
    "lowest",
    "average",
    "mean",
    "median",
    "sum",
    "count",
    "histogram",
    "plot",
    "chart",
    "table",
    " by ",
    "топ",
    "средн",
    "посчитай",
    "построй",
    "таблиц",
)

TEMPLATE_PHRASES = (
    "widest value band",
    "group is not internally uniform",
    "evidence weaker for sparse groups",
    "analytical picture is unchanged",
    "subgroup mix",
)

LEAKAGE_MARKERS = (
    "the active comparison is still",
    "the current evidence centers on",
    "judge it by the leading",
    "next supporting evidence",
    "supporting evidence must",
    "continuation",
    "artifact routing",
    "branch state",
    "branch-routing",
    "planner",
    "executor",
    "fallback mode",
    "fallback",
    "orchestration",
    "hidden routing",
)

WEAK_METRIC_NAMES = {
    "row id",
    "row_id",
    "id",
    "identifier",
    "sum",
    "count",
    "index",
    "postal code",
    "postcode",
    "zip",
    "zip code",
}

BUSINESS_LENSES = {
    "risk_analysis": ("concentration", "dependency", "consistency", "stability", "exposure"),
    "customer_behavior_analysis": ("distribution", "consistency", "segmentation", "dominance", "coverage"),
    "executive_summary": ("priority", "impact", "readiness", "evidence strength", "next validation"),
    "strategic_recommendation": ("leverage", "impact", "effort", "upside", "risk reduction"),
    "chart_interpretation": ("leader stability", "segment contrast", "outlier effect", "sample size", "relevance"),
    "anomaly_analysis": ("outlier effect", "data quality", "exception", "exposure", "investigation priority"),
}


def classify_reasoning_intent(question: str) -> ReasoningIntent:
    text = _norm(question)
    for mode, markers in INTERPRETIVE_MODES:
        if any(marker in text for marker in markers):
            return ReasoningIntent(kind="interpretive", mode=mode, requires_llm=True)
    if any(marker in text for marker in HYBRID_MARKERS):
        return ReasoningIntent(kind="hybrid", mode="anomaly_analysis" if "outlier" in text else "chart_interpretation", requires_llm=True)
    if any(marker in text for marker in ("which of them", "which is most", "most dangerous", "prioritize", "rank these", "какой из них", "самый опас")):
        return ReasoningIntent(kind="interpretive", mode="strategic_recommendation", requires_llm=True)
    if any(marker in text for marker in COMPUTATIONAL_MARKERS):
        return ReasoningIntent(kind="computational", mode="computed_analysis", requires_llm=False)
    return ReasoningIntent(kind="computational", mode="computed_analysis", requires_llm=False)


def apply_llm_reasoning_layer(
    *,
    question: str,
    output: dict[str, Any],
    df: Any = None,
    data_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Cross-dataset outputs already have structured findings; do not apply
    # single-dataset concentration/PM reasoning on top of them.
    trace = output.get("trace_metadata") if isinstance(output.get("trace_metadata"), dict) else {}
    if trace.get("analysis_type") in ("cross_dataset", "cross_dataset_output", "structural"):
        return output
    intent = classify_reasoning_intent(question)
    if not intent.requires_llm or not isinstance(output, dict):
        return output
    evidence = build_reasoning_evidence(question=question, output=output, df=df, data_context=data_context, intent=intent)
    try:
        llm = make_llm("analytical_reasoning")
        response = llm.invoke(_reasoning_prompt(evidence))
        text = _response_text(response)
    except Exception as exc:
        local = _local_grounded_synthesis(evidence, intent)
        if local:
            output = _merge_reasoning_output(output, local, intent, evidence)
            output.setdefault("trace_metadata", {})
            if isinstance(output["trace_metadata"], dict):
                output["trace_metadata"]["llm_reasoning_status"] = "local_synthesis"
                output["trace_metadata"]["llm_reasoning_mode"] = intent.mode
            _append_timeline(output, {"tool": "llm_analytical_reasoning", "status": "skipped", "reason": str(exc)[:180], "mode": intent.mode})
            return output
        _append_timeline(output, {"tool": "llm_analytical_reasoning", "status": "skipped", "reason": str(exc)[:180], "mode": intent.mode})
        output.setdefault("trace_metadata", {})
        if isinstance(output["trace_metadata"], dict):
            output["trace_metadata"]["llm_reasoning_status"] = "skipped"
            output["trace_metadata"]["llm_reasoning_mode"] = intent.mode
        return output
    polished = _quality_gate_text(_clean_llm_text(text), evidence, intent)
    if not polished:
        return output
    return _merge_reasoning_output(output, polished, intent, evidence)


def build_reasoning_evidence(
    *,
    question: str,
    output: dict[str, Any],
    df: Any = None,
    data_context: dict[str, Any] | None = None,
    intent: ReasoningIntent | None = None,
) -> dict[str, Any]:
    structured = output.get("structured_report") if isinstance(output.get("structured_report"), dict) else {}
    artifacts = output.get("artifacts") if isinstance(output.get("artifacts"), list) else []
    return {
        "question": question,
        "intent": (intent or classify_reasoning_intent(question)).__dict__,
        "computed_summary": str(output.get("summary") or output.get("final_answer") or structured.get("summary") or ""),
        "computed_findings": _string_list(output.get("key_findings") or structured.get("key_findings")),
        "evidence": _string_list(output.get("evidence") or structured.get("evidence")),
        "limitations": _string_list(output.get("limitations") or structured.get("limitations")),
        "artifacts": [_artifact_brief(item) for item in artifacts[:5] if isinstance(item, dict)],
        "data_profile": _data_profile(df),
        "conversation_context": _conversation_brief(data_context),
    }


def _reasoning_prompt(evidence: dict[str, Any]) -> str:
    mode = (evidence.get("intent") or {}).get("mode") if isinstance(evidence.get("intent"), dict) else ""
    return (
        "You are an evidence-grounded business analytics assistant.\n"
        "Use the computed evidence below. Do not invent metrics, columns, causal relationships, joins, or trends.\n"
        "Write the final answer only, in the user's language when clear from the question.\n"
        "Start with the main conclusion and rank the most important finding first.\n"
        "Connect the computed facts to practical implications when supported by the evidence.\n"
        "Separate primary risk/opportunity from secondary observations. Mention weak evidence as weak; use stronger language when the computed evidence is strong.\n"
        "Avoid robotic phrases such as 'widest value band', 'group not internally uniform', 'analytical picture unchanged', and ontology/template dumps.\n"
        "Use cautious wording such as 'suggests', 'may indicate', or 'could reflect' for non-causal evidence, but do not over-hedge every sentence.\n"
        f"Reasoning mode: {mode or 'business_analysis'}.\n\n"
        f"Evidence package:\n{json.dumps(evidence, ensure_ascii=False, default=str, indent=2)}"
    )


def _merge_reasoning_output(output: dict[str, Any], text: str, intent: ReasoningIntent, evidence: dict[str, Any]) -> dict[str, Any]:
    merged = dict(output)
    text = sanitize_user_visible_text(text)
    merged["summary"] = text
    merged["final_answer"] = text
    structured = dict(merged.get("structured_report") if isinstance(merged.get("structured_report"), dict) else {})
    structured["summary"] = text
    structured["key_findings"] = _reasoning_findings(text)
    structured["evidence"] = evidence.get("evidence") or evidence.get("computed_findings") or ["Computed evidence was interpreted by the LLM reasoning layer."]
    structured["limitations"] = evidence.get("limitations") or ["Interpretation is grounded in computed evidence and does not establish causality."]
    structured.setdefault("next_steps", _default_next_steps(intent.mode))
    merged["structured_report"] = structured
    merged["key_findings"] = structured["key_findings"]
    merged["limitations"] = structured["limitations"]
    merged["next_steps"] = structured["next_steps"]
    _append_timeline(merged, {"tool": "llm_analytical_reasoning", "status": "ok", "mode": intent.mode, "intent": intent.kind})
    merged.setdefault("trace_metadata", {})
    if isinstance(merged["trace_metadata"], dict):
        merged["trace_metadata"]["llm_reasoning_status"] = "ok"
        merged["trace_metadata"]["llm_reasoning_mode"] = intent.mode
        merged["trace_metadata"]["llm_reasoning_intent"] = intent.kind
    return merged


def sanitize_user_visible_output(output: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(output, dict):
        return output
    sanitized = dict(output)
    for key in ("summary", "final_answer", "result_preview"):
        if key in sanitized and isinstance(sanitized.get(key), str):
            sanitized[key] = sanitize_user_visible_text(str(sanitized[key]))
    structured = sanitized.get("structured_report")
    if isinstance(structured, dict):
        structured = dict(structured)
        for key in ("summary", "answer", "question"):
            if isinstance(structured.get(key), str):
                structured[key] = sanitize_user_visible_text(str(structured[key]))
        for key in ("key_findings", "evidence", "limitations", "next_steps"):
            if isinstance(structured.get(key), list):
                structured[key] = [sanitize_user_visible_text(str(item)) for item in structured[key]]
        sanitized["structured_report"] = structured
    for key in ("key_findings", "evidence", "limitations", "next_steps"):
        if isinstance(sanitized.get(key), list):
            sanitized[key] = [sanitize_user_visible_text(str(item)) for item in sanitized[key]]
    return sanitized


def sanitize_user_visible_text(text: str) -> str:
    cleaned_lines: list[str] = []
    for line in str(text or "").splitlines():
        normalized = _norm(line)
        if any(marker in normalized for marker in LEAKAGE_MARKERS):
            replacement = _leakage_replacement(normalized)
            if replacement and (not cleaned_lines or cleaned_lines[-1] != replacement):
                cleaned_lines.append(replacement)
            continue
        cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines).strip()
    return _reduce_repetitive_language(cleaned)


def _data_profile(df: Any) -> dict[str, Any]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {}
    numeric = list(df.select_dtypes(include="number").columns[:8])
    categorical = [column for column in df.columns if column not in numeric][:8]
    profile: dict[str, Any] = {"rows": int(len(df)), "columns": int(len(df.columns)), "numeric_columns": numeric, "categorical_columns": categorical}
    if numeric:
        desc = df[numeric].describe().round(3).to_dict()
        profile["numeric_summary"] = desc
    return profile


def _artifact_brief(item: dict[str, Any]) -> dict[str, Any]:
    content = item.get("content")
    rows = []
    if isinstance(content, list):
        rows = content[:5]
    elif isinstance(content, dict) and isinstance(content.get("rows"), list):
        rows = content["rows"][:5]
    return {
        "type": item.get("artifact_type") or item.get("type"),
        "title": item.get("title"),
        "metadata": item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
        "sample_rows": rows,
    }


def _conversation_brief(data_context: dict[str, Any] | None) -> dict[str, Any]:
    context = data_context.get("conversation_context") if isinstance(data_context, dict) else {}
    if not isinstance(context, dict):
        return {}
    return {
        "latest_findings": context.get("latest_findings", [])[:5] if isinstance(context.get("latest_findings"), list) else [],
        "latest_report_summary": context.get("latest_report_summary", ""),
        "artifact_titles": context.get("artifact_titles", [])[:8] if isinstance(context.get("artifact_titles"), list) else [],
    }


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return " ".join(str(item.get("text") if isinstance(item, dict) else item) for item in content)
    return str(content or "")


def _clean_llm_text(text: str) -> str:
    cleaned = "\n".join(line.rstrip() for line in str(text or "").strip().splitlines()).strip()
    for phrase in TEMPLATE_PHRASES:
        cleaned = cleaned.replace(phrase, _template_replacement(phrase))
    return cleaned


def _local_grounded_synthesis(evidence: dict[str, Any], intent: ReasoningIntent) -> str:
    """Evidence-restating fallback when LLM reasoning is unavailable.

    Instead of inserting generic business filler, restate the computed findings
    and let the evidence speak for itself.
    """
    summary = str(evidence.get("computed_summary") or "").strip()
    findings = evidence.get("computed_findings") if isinstance(evidence.get("computed_findings"), list) else []
    data_profile = evidence.get("data_profile") if isinstance(evidence.get("data_profile"), dict) else {}
    artifacts = evidence.get("artifacts") if isinstance(evidence.get("artifacts"), list) else []
    concentration = _concentration_signal(artifacts)
    metric = _business_metric_label(concentration.get("metric") or _first_metric(data_profile))
    metric_label = f"`{metric}`" if metric else "the primary measure"

    # Build an evidence-grounded response from computed facts
    parts = []
    if summary:
        parts.append(summary)
    if findings:
        for i, finding in enumerate(findings[:3], 1):
            parts.append(f"{i}. {finding}")
    if concentration:
        leader = concentration.get("leader")
        share = concentration.get("share")
        if leader and share:
            parts.append(f"Concentration note: `{leader}` accounts for {share} of {metric_label}.")
    if not parts:
        return ""
    parts.append("Validate with an outlier check and a segment-level breakdown before drawing conclusions.")
    return "\n\n".join(parts)


def _quality_gate_text(text: str, evidence: dict[str, Any], intent: ReasoningIntent) -> str:
    sanitized = sanitize_user_visible_text(text)
    normalized = _norm(sanitized)
    if _too_repetitive(normalized) or _too_much_statistical_filler(normalized) or _too_long(sanitized):
        local = _local_grounded_synthesis(evidence, intent)
        return sanitize_user_visible_text(local or sanitized)
    return sanitized


def _leakage_replacement(normalized_line: str) -> str:
    if "the active comparison is still" in normalized_line or "the current evidence centers on" in normalized_line or "judge it by the leading" in normalized_line:
        return ""
    if "next supporting evidence" in normalized_line or "supporting evidence must" in normalized_line:
        return "The next useful check is to validate the current metric with a concrete chart, table, or quality check."
    if any(marker in normalized_line for marker in ("planner", "executor", "fallback", "branch state", "artifact routing", "orchestration", "continuation")):
        return ""
    return ""


def _reduce_repetitive_language(text: str) -> str:
    replacements = {
        "subgroup mix": "segment composition",
        "wide spread": "uneven performance",
        "widest spread": "largest difference",
        "value spread": "variation",
        "forecasting risk": "planning risk",
    }
    cleaned = text
    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target).replace(source.title(), target.title())
    return cleaned.strip()


def _template_replacement(phrase: str) -> str:
    if phrase == "subgroup mix":
        return "segment composition"
    if phrase == "analytical picture is unchanged":
        return "the main conclusion remains the same"
    return "variation"


def _too_repetitive(normalized: str) -> bool:
    watched = ("spread", "variability", "subgroup", "forecasting risk", "may indicate", "could reflect")
    return any(normalized.count(term) > 1 for term in watched)


def _too_much_statistical_filler(normalized: str) -> bool:
    return sum(normalized.count(term) for term in ("spread", "range", "variance", "uniform", "subgroup")) >= 4


def _too_long(text: str) -> bool:
    return len(str(text).split()) > 180


def _concentration_signal(artifacts: list[Any]) -> dict[str, Any]:
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        rows = artifact.get("sample_rows")
        metadata = artifact.get("metadata") if isinstance(artifact.get("metadata"), dict) else {}
        if not isinstance(rows, list) or len(rows) < 2:
            continue
        value_key = next((key for key in ("total", "sum", "mean", "value", metadata.get("metric")) if key and isinstance(rows[0], dict) and key in rows[0]), None)
        if not value_key:
            continue
        values = []
        for row in rows:
            if isinstance(row, dict):
                try:
                    values.append(abs(float(row.get(value_key) or 0)))
                except (TypeError, ValueError):
                    pass
        if len(values) >= 2 and values[0] >= max(sum(values), 1.0) * 0.45:
            return {"metric": metadata.get("metric") or value_key, "top_share": values[0] / max(sum(values), 1.0)}
    return {}


def _first_metric(data_profile: dict[str, Any]) -> str:
    metrics = data_profile.get("numeric_columns") if isinstance(data_profile, dict) else []
    if not isinstance(metrics, list):
        return ""
    for metric in metrics:
        label = _business_metric_label(metric)
        if label:
            return label
    return ""


def _business_metric_label(metric: Any) -> str:
    value = str(metric or "").strip()
    normalized = _norm(value)
    if not value or normalized in WEAK_METRIC_NAMES or normalized.endswith(" id") or normalized.endswith(" code"):
        return ""
    if normalized == "quantity":
        return "transaction volume"
    return value


def _reasoning_findings(text: str) -> list[str]:
    lines = [line.strip(" -•\t") for line in text.splitlines() if line.strip()]
    useful = [line for line in lines if len(line) > 20]
    return useful[:5] or [text[:240]]


def _default_next_steps(mode: str) -> list[str]:
    if mode == "risk_analysis":
        return ["Quantify the largest suspected risk with a focused metric, segment, and outlier check."]
    if mode == "chart_interpretation":
        return ["Validate the chart interpretation by checking sample sizes, outliers, and the strongest segment split."]
    return ["Validate the interpretation with a focused follow-up calculation or chart."]


def _append_timeline(output: dict[str, Any], event: dict[str, Any]) -> None:
    timeline = output.setdefault("tool_timeline", [])
    if isinstance(timeline, list):
        timeline.append(event)
    structured = output.get("structured_report")
    if isinstance(structured, dict):
        nested = structured.setdefault("tool_timeline", [])
        if isinstance(nested, list):
            nested.append(event)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item).strip()]
    if value:
        return [str(value)]
    return []


def _norm(value: Any) -> str:
    return " ".join(str(value or "").casefold().replace("_", " ").split())
