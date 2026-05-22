from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from source.product.analytical_graph import has_impossible_counts
from source.product.evidence_resolution import transformation_state_from_payload
from source.product.language_policy import ResponseLanguagePolicy


@dataclass(frozen=True)
class ConversationResponse:
    text: str
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    updated_state: dict[str, Any] = field(default_factory=dict)
    response_kind: str = "conversation_engine"


def answer_from_conversation_state(
    *,
    question: str,
    conversation_context: dict[str, Any] | None,
    recent_artifacts: list[Any] | None = None,
) -> ConversationResponse | None:
    context = conversation_context if isinstance(conversation_context, dict) else {}
    state = context.get("conversation_state") if isinstance(context.get("conversation_state"), dict) else {}
    latest_findings = [str(item) for item in (context.get("latest_findings") or []) if str(item).strip()]
    artifacts = list(recent_artifacts or [])
    metric = str(state.get("active_metric") or "").strip()
    dimension = str(state.get("active_dimension") or "").strip()
    latest_transformation = _latest_transformation_from_state(state) or _latest_transformation_artifacts(artifacts)
    intent = route_conversation_intent(question, has_active_context=bool(metric or dimension or latest_transformation))

    chart_explanation = _chart_explanation_response(question, artifacts, state, context)
    if chart_explanation:
        return chart_explanation
    if latest_transformation and _is_ranking_followup(question):
        if _is_change_followup(question):
            return _post_transformation_impact(question, metric or str(latest_transformation.get("metric") or ""), dimension or str(latest_transformation.get("dimension") or ""), latest_transformation)
        return _post_transformation_leaders(question, metric or str(latest_transformation.get("metric") or ""), dimension or str(latest_transformation.get("dimension") or ""), latest_transformation)
    if intent == "post_transformation_followup" and latest_transformation:
        return _post_transformation_leaders(question, metric, dimension, latest_transformation)
    if intent == "hypothesis_request":
        return _hypothesis_response(question, metric, dimension, latest_transformation)
    if intent == "validation_request":
        hypothesis_finding = next((item for item in reversed(latest_findings) if "гипотез" in _normalize(item) or "hypothesis" in _normalize(item)), "")
        if hypothesis_finding:
            state = {**state, "active_hypothesis": hypothesis_finding}
        elif not state.get("active_hypothesis") and latest_findings:
            state = {**state, "active_hypothesis": latest_findings[-1]}
        return _validation_response(question, metric, dimension, latest_transformation, state)
    return None


def _chart_explanation_response(question: str, artifacts: list[Any], state: dict[str, Any], context: dict[str, Any]) -> ConversationResponse | None:
    action = context.get("active_message_metadata") if isinstance(context.get("active_message_metadata"), dict) else {}
    if action.get("action") != "explain_artifact" and not _is_chart_explanation_question(question):
        return None
    active_artifact_id = str(action.get("artifact_id") or state.get("active_artifact_id") or "").strip()
    strict_artifact = action.get("action") == "explain_artifact" and bool(active_artifact_id)
    chart = _resolve_chart_artifact(question, artifacts, artifact_id=active_artifact_id)
    if not chart:
        if strict_artifact:
            text = f"I cannot explain this chart because the exact artifact `{active_artifact_id}` is not available in the current artifact payload."
            return ConversationResponse(text=text, findings=[text], response_kind="chart_explanation")
        return None
    content = _content(chart)
    metadata = _metadata(chart)
    rows = _artifact_rows(chart)
    metric = str(metadata.get("metric") or content.get("metric") or state.get("active_metric") or "").strip()
    dimension = str(metadata.get("dimension") or content.get("dimension") or content.get("x") or state.get("active_dimension") or "").strip()
    aggregation = str(metadata.get("aggregation") or content.get("aggregation") or content.get("y") or state.get("active_chart_aggregation") or "").strip()
    filters = metadata.get("filters") if isinstance(metadata.get("filters"), list) else content.get("filters") if isinstance(content.get("filters"), list) else []
    chart_type = str(metadata.get("chart_type") or content.get("chart_type") or "").strip()
    title = _title(chart) or (f"{metric} by {dimension}" if metric and dimension else "the chart")
    has_comparison_groups = isinstance(content.get("comparison_groups"), list) and bool(content.get("comparison_groups"))
    if not rows and not has_comparison_groups:
        text = f"`{title}` is saved as a {chart_type or 'chart'}, but it does not include displayed values to explain yet."
        return ConversationResponse(text=text, findings=[text], response_kind="chart_explanation")
    value_key = _artifact_value_key(rows[0], aggregation) if rows else "count"
    aggregation_label = _aggregation_label(aggregation or value_key)
    filter_text = _filter_phrase(filters)
    transformation_type = str(metadata.get("transformation_type") or content.get("transformation_type") or state.get("active_transformation") or "").strip()
    if transformation_type:
        text = _transformed_chart_explanation(title, metric, dimension, rows, value_key, aggregation_label, metadata, content)
    elif chart_type == "histogram" or content.get("visualization_type") == "histogram":
        text = _histogram_explanation(title, metric, dimension, rows, filter_text, content)
    elif chart_type == "line" or content.get("time_axis") or content.get("timestamp"):
        text = _temporal_chart_explanation(title, metric, content, rows, value_key, aggregation_label, filter_text)
    elif chart_type == "heatmap":
        text = _heatmap_explanation(title, metric, rows, value_key)
    else:
        text = _grouped_chart_explanation(title, metric, dimension, rows, value_key, aggregation_label, filter_text)
    return ConversationResponse(
        text=text,
        findings=[text],
        updated_state={
            "active_metric": metric,
            "active_dimension": dimension,
            "active_chart": {"title": title, "chart_type": chart_type, "metric": metric, "dimension": dimension, "aggregation": aggregation},
            "active_chart_aggregation": aggregation,
            "last_successful_analysis_type": "chart_explanation",
            "last_successful_answer_summary": text[:500],
        },
        response_kind="chart_explanation",
    )


def _histogram_explanation(title: str, metric: str, dimension: str, rows: list[dict[str, Any]], filter_text: str, content: dict[str, Any] | None = None) -> str:
    content = content or {}
    comparison_groups = content.get("comparison_groups") if isinstance(content.get("comparison_groups"), list) else []
    if comparison_groups:
        parts = []
        labels = []
        for group in comparison_groups:
            if not isinstance(group, dict):
                continue
            label = str(group.get("label") or group.get("group") or "group")
            labels.append(label)
            bins = [item for item in group.get("bins", []) if isinstance(item, dict)] if isinstance(group.get("bins"), list) else []
            counts = [int(item.get("count") or 0) for item in bins]
            total = int(group.get("row_count") or group.get("record_count") or group.get("n") or sum(counts))
            if not counts or total <= 0:
                continue
            peak_idx = max(range(len(counts)), key=lambda idx: counts[idx])
            peak = bins[peak_idx]
            mean = float(group.get("mean") or 0)
            median = float(group.get("median") or 0)
            skew = "right-skewed" if mean > median else "left-skewed" if mean < median else "centered"
            parts.append(f"`{label}` peaks at `{peak.get('label')}` with {counts[peak_idx]} records and is {skew} by mean vs median")
        group_text = " vs ".join(f"`{label}`" for label in labels[:4])
        if parts:
            return (
                f"`{title}` is a histogram comparison for `{metric}` across {group_text}. "
                + "; ".join(parts)
                + ". Read it through distribution overlap, spread, tails, and center shift rather than grouped rankings."
            )
    if rows and {"mean", "median", "min", "max", "n"} <= set(rows[0]):
        bits = []
        for row in rows[:4]:
            label = row.get(dimension) or row.get("label") or row.get("series") or "group"
            mean = float(row.get("mean") or 0)
            median = float(row.get("median") or 0)
            spread = float(row.get("max") or 0) - float(row.get("min") or 0)
            skew = "right-skewed" if mean > median else "left-skewed" if mean < median else "balanced around the center"
            bits.append(f"`{label}` has median {median:.2f}, mean {mean:.2f}, spread {spread:.2f}, so it is {skew}")
        if len(rows) >= 2:
            first, second = rows[0], rows[1]
            overlap = "overlap" if float(first.get("max") or 0) >= float(second.get("min") or 0) and float(second.get("max") or 0) >= float(first.get("min") or 0) else "limited overlap"
            return f"`{title}` compares the distribution of `{metric}`{filter_text}. " + "; ".join(bits) + f". The ranges show {overlap}, so the comparison should be read through spread and tails, not only averages."
        return f"`{title}` shows the distribution of `{metric}`{filter_text}. " + "; ".join(bits) + "."
    counts = [int(row.get("count") or row.get("record_count") or 0) for row in rows]
    labels = [str(row.get("bin") or row.get("label") or "") for row in rows]
    total = sum(counts)
    if not counts or total <= 0:
        return f"`{title}` shows a `{metric}` distribution{filter_text}, but the bins do not contain enough count information to describe its shape."
    peak_idx = max(range(len(counts)), key=lambda idx: counts[idx])
    first_count = counts[0]
    last_count = counts[-1]
    tail = "heavier upper tail" if last_count > first_count else "heavier lower tail" if first_count > last_count else "balanced tails"
    concentration = counts[peak_idx] / total * 100.0
    return (
        f"`{title}` shows the distribution of `{metric}`{filter_text}. "
        f"The densest bin is `{labels[peak_idx]}` with {counts[peak_idx]} records ({concentration:.1f}% of the displayed rows). "
        f"The tails are {tail}, and the full bin range indicates the spread and possible outliers."
    )


def _transformed_chart_explanation(
    title: str,
    metric: str,
    dimension: str,
    rows: list[dict[str, Any]],
    value_key: str,
    aggregation_label: str,
    metadata: dict[str, Any],
    content: dict[str, Any],
) -> str:
    transformation_type = str(metadata.get("transformation_type") or content.get("transformation_type") or "adjustment")
    transformation_label = transformation_type.replace("_", " ")
    adjusted_rows = rows[:5]
    leaders = ", ".join(
        f"`{row.get(dimension) or row.get('label')}` ({float(row.get(value_key) or row.get('adjusted_mean') or row.get('mean') or 0):.2f})"
        for row in adjusted_rows
    )
    impact = metadata.get("transformation_impact") if isinstance(metadata.get("transformation_impact"), dict) else content.get("transformation_impact") if isinstance(content.get("transformation_impact"), dict) else {}
    comparison_rows = impact.get("comparison_rows") if isinstance(impact.get("comparison_rows"), list) else []
    dropped = [
        row
        for row in comparison_rows
        if isinstance(row, dict) and (float(row.get("rank_change") or 0) >= 3 or float(row.get("adjusted_rank") or 0) > float(row.get("original_rank") or 0) + 2)
    ][:3]
    stable = [
        row
        for row in comparison_rows
        if isinstance(row, dict) and float(row.get("original_rank") or 0) <= 5 and float(row.get("adjusted_rank") or 0) <= 5
    ][:3]
    parts = [
        f"`{title}` explains `{metric}` by `{dimension}` after {transformation_label}.",
        f"The adjusted leaders are {leaders}.",
    ]
    if dropped:
        parts.append("Groups that weaken most after the adjustment: " + _shift_phrase(dropped, dimension) + ".")
    if stable:
        parts.append("Groups that remain near the top: " + _shift_phrase(stable, dimension) + ".")
    parts.append(f"Read this as an adjusted {aggregation_label} view: the ranking is about what remains after the transformation, not the raw unfiltered chart.")
    return " ".join(parts)


def _temporal_chart_explanation(title: str, metric: str, content: dict[str, Any], rows: list[dict[str, Any]], value_key: str, aggregation_label: str, filter_text: str) -> str:
    values = [float(row.get(value_key) or row.get("value") or 0) for row in rows if isinstance(row, dict)]
    time_axis = content.get("time_axis") or content.get("timestamp") or "time"
    if len(values) >= 2:
        direction = "rises" if values[-1] > values[0] else "declines" if values[-1] < values[0] else "stays flat"
        changes = [values[idx] - values[idx - 1] for idx in range(1, len(values))]
        volatility = sum(abs(item) for item in changes) / len(changes) if changes else 0.0
        spike = max(changes, key=abs) if changes else 0.0
        return (
            f"`{title}` shows {aggregation_label} `{metric}` over `{time_axis}`{filter_text}. "
            f"The series {direction} overall, from {values[0]:.2f} to {values[-1]:.2f}; average period-to-period movement is {volatility:.2f}, with the largest swing at {spike:+.2f}."
        )
    return f"`{title}` shows {aggregation_label} `{metric}` over `{time_axis}`{filter_text}, but only one period is available."


def _heatmap_explanation(title: str, metric: str, rows: list[dict[str, Any]], value_key: str) -> str:
    ordered = sorted(rows, key=lambda row: float(row.get(value_key) or row.get("value") or 0), reverse=True)
    leaders = ", ".join(
        "`" + " / ".join(str(row.get(key)) for key in ("year", "month") if row.get(key) not in (None, "")) + f"` ({float(row.get(value_key) or row.get('value') or 0):.2f})"
        for row in ordered[:3]
    )
    return f"`{title}` highlights concentration zones for `{metric}`. The strongest cells are {leaders}; read this as a periodicity and cluster view rather than a simple ranking."


def _grouped_chart_explanation(title: str, metric: str, dimension: str, rows: list[dict[str, Any]], value_key: str, aggregation_label: str, filter_text: str) -> str:
    leaders = rows[:3]
    laggards = rows[-3:] if len(rows) >= 3 else []
    leader_bits = ", ".join(
        f"`{row.get(dimension) or row.get('period') or row.get('label')}` ({float(row.get(value_key) or 0):.2f})"
        for row in leaders
    )
    laggard_bits = ", ".join(
        f"`{row.get(dimension) or row.get('period') or row.get('label')}` ({float(row.get(value_key) or 0):.2f})"
        for row in laggards
    )
    gap = 0.0
    if len(rows) >= 2:
        gap = float(rows[0].get(value_key) or 0) - float(rows[1].get(value_key) or 0)
    tail = f" Lowest displayed groups: {laggard_bits}." if laggard_bits else ""
    return (
        f"`{title}` ranks `{dimension}` by {aggregation_label} `{metric}`{filter_text}. "
        f"{leader_bits} lead the displayed values; the top-two gap is {gap:.2f}.{tail} "
        f"The main takeaway is the size of the ranking gap and how concentrated `{metric}` is among the leading groups."
    )


def clarification_from_state(conversation_context: dict[str, Any] | None) -> ConversationResponse | None:
    context = conversation_context if isinstance(conversation_context, dict) else {}
    state = context.get("conversation_state") if isinstance(context.get("conversation_state"), dict) else {}
    metric = str(state.get("active_metric") or "").strip()
    dimension = str(state.get("active_dimension") or "").strip()
    if not (metric or dimension):
        return None
    topic = _topic(metric, dimension)
    return ConversationResponse(
        text=f"Do you want to continue the current {topic} branch or start a new check? I will not reset the analysis to an overview unless you ask for that explicitly.",
        response_kind="clarification_needed",
    )


def route_conversation_intent(question: str, *, has_active_context: bool) -> str:
    text = _normalize(question)
    if any(marker in text for marker in ("overview", "describe dataset", "что ты можешь сказать", "опиши данные")):
        return "overview_request"
    if _is_hypothesis_followup_question(text):
        return "validation_request"
    if any(marker in text for marker in ("hypothesize", "сформируй hypothesis", "сформулируй hypothesis")):
        return "hypothesis_request"
    if text.startswith(("hypothesis:", "hypothesis -")) or "hypothesis:" in text:
        return "unknown"
    if any(marker in text for marker in ("как это проверить", "how to validate", "how would you validate", "how can we validate", "how can we test", "validate this", "validate it", "validate further", "проверить", "доказать", "подтвердить")):
        return "validation_request"
    transformation_reference = any(marker in text for marker in ("остаются", "остались", "после", "after", "remain", "remains", "лидерами", "лидеры", "просели", "dropped", "shift"))
    if has_active_context and transformation_reference:
        return "post_transformation_followup"
    if any(marker in text for marker in ("shipping", "ship ", "ship mode", "delivery", "достав", "shipping behavior")):
        return "unknown"
    if has_active_context and any(marker in text for marker in ("это", "этот", "эта", "these", "this", "that")):
        return "clarification_needed"
    return "unknown"


def _is_ranking_followup(question: str) -> bool:
    text = _normalize(question)
    return any(
        marker in text
        for marker in (
            "strongest",
            "leaders",
            "leader",
            "top",
            "best",
            "remain",
            "remains",
            "what changed",
            "changed",
            "самые сильные",
            "сильн",
            "лидер",
            "остаются",
            "остались",
            "что измен",
        )
    )


def _is_change_followup(question: str) -> bool:
    text = _normalize(question)
    return any(marker in text for marker in ("what changed", "changed", "change after", "after filtering", "findings became unreliable", "became unreliable", "что измен", "какие findings"))


def overview_forbidden(question: str, conversation_context: dict[str, Any] | None) -> bool:
    context = conversation_context if isinstance(conversation_context, dict) else {}
    state = context.get("conversation_state") if isinstance(context.get("conversation_state"), dict) else {}
    has_active = bool(state.get("active_metric") or state.get("active_dimension") or state.get("active_chart"))
    if not has_active:
        return False
    return route_conversation_intent(question, has_active_context=True) != "overview_request"


def response_quality_gate(
    *,
    question: str,
    response_text: str,
    conversation_context: dict[str, Any] | None,
) -> tuple[bool, str]:
    text = _normalize(response_text)
    if not text:
        return False, "empty_response"
    if has_impossible_counts(response_text):
        return False, "impossible_counts"
    context = conversation_context if isinstance(conversation_context, dict) else {}
    state = context.get("conversation_state") if isinstance(context.get("conversation_state"), dict) else {}
    metric = str(state.get("active_metric") or "").strip()
    dimension = str(state.get("active_dimension") or "").strip()
    branch_type = str(state.get("active_branch_type") or "").strip()
    active_target = state.get("active_analytical_target") if isinstance(state.get("active_analytical_target"), dict) else {}
    target_dimension = str(active_target.get("dimension") or dimension).strip()
    target_metric = str(active_target.get("metric") or metric).strip()
    target_mechanism = str(active_target.get("active_mechanism") or "").strip()
    time_axis = str(state.get("active_time_axis") or "").strip()
    overview_markers = (
        "dataset has",
        "quantitative fields",
        "основу для аналитического расследования",
        "количественные точки входа",
        "для сегментации",
    )
    if overview_forbidden(question, conversation_context) and any(marker in text for marker in overview_markers):
        return False, "overview_locked_out"
    question_text = _normalize(question)
    if _wrong_response_language(question, response_text, state):
        return False, "wrong_response_language"
    system_leakage = (
        "i do not have a new computed result",
        "saved finding",
        "move beyond the repeated summary",
        "the active target is",
        "the contradiction check should stay",
        "the evidence supports",
        "the evidence is a ",
        "the evidence is the ",
        "the saved evidence",
        "switching back to the",
    )
    if any(marker in text for marker in system_leakage):
        return False, "framework_or_backend_leakage"
    ungrounded_execution_filler = (
        "likely to vary",
        "strongest groups will",
        "will be the ones with",
        "the analytical picture is unchanged",
        "the useful extension is a direct comparison",
        "small samples and extreme values can make a group look stronger",
    )
    if any(marker in text for marker in ungrounded_execution_filler):
        return False, "ungrounded_execution_filler"
    if any(marker in question_text for marker in ("evidence", "support", "подтверж", "доказ")) and "hypothesis" in text and ("limitation:" in text or "next validation:" in text or "confidence is" in text):
        return False, "repeated_full_hypothesis_answer"
    if any(marker in question_text for marker in ("evidence", "support", "contradict", "counterevidence", "refute", "подтверж", "доказ", "противореч", "опроверг")) and active_target:
        if any(marker in text for marker in ("the evidence is the active", "the evidence is grounded", "the counterevidence should", "it would contradict")):
            return False, "generic_evidence_or_contradiction"
        if target_dimension and f"`{target_dimension.casefold()}`" not in text and target_dimension.casefold() not in text:
            return False, "evidence_ignored_active_target"
        if target_mechanism and target_mechanism not in text and target_mechanism.replace("_", " ") not in text:
            mechanism_words = [part for part in target_mechanism.split("_") if len(part) >= 5]
            mechanism_ok = any(part in text for part in mechanism_words)
            if target_mechanism == "operational_effect" and any(part in text for part in ("delay", "shipping", "ship mode")):
                mechanism_ok = True
            if target_mechanism == "sparse_group_instability" and any(part in text for part in ("n=", "sample", "sparse")):
                mechanism_ok = True
            if target_mechanism in {"outlier_concentration", "outlier_effect", "concentration", "large_record_dependence"} and any(part in text for part in ("rank #", "collapsed", "extreme", "large-record")):
                mechanism_ok = True
            if not mechanism_ok:
                return False, "evidence_ignored_active_mechanism"
    if _is_quality_question(question_text) and _looks_like_grouped_country_answer(text):
        return False, "quality_answer_used_grouped_country"
    if _is_shipping_behavior_question(question_text) and _looks_like_raw_ship_date_grouping(text):
        return False, "shipping_used_raw_ship_date"
    if _contradicts_active_transformation(text, state):
        return False, "contradicts_active_transformation"
    if _asks_transformation_change(question_text) and _looks_like_adjusted_leaders_only(text):
        return False, "shallow_transformation_impact"
    if _is_affected_findings_question(question_text) and _looks_generic_affected_findings(text):
        return False, "generic_affected_findings"
    if any(marker in question_text for marker in ("city", "cities", "город")) and "`country`" in text:
        return False, "ignored_explicit_city"
    if "standard class" in question_text and "`country`" in text:
        return False, "ignored_category_value"
    if _is_hypothesis_question(question_text):
        if "the active conclusion is" in text:
            return False, "generic_hypothesis_filler"
        if _looks_like_shallow_hypothesis(text):
            return False, "shallow_hypothesis_verdict"
        if any(marker in text for marker in ("гипотеза: различия", "checking it requires", "проверять ее нужно")):
            return False, "generic_hypothesis_template"
        if "technology" in question_text and "`segment`" in text and "`category`" not in text:
            return False, "ignored_explicit_hypothesis_entity"
        if "standard class" in question_text and "`ship mode`" not in text:
            return False, "ignored_explicit_hypothesis_value"
        if any(marker in question_text for marker in ("city", "cities", "город")) and "`city`" not in text:
            return False, "ignored_hypothesis_dimension"
        if "outlier" in question_text and "outlier" not in text:
            return False, "ignored_hypothesis_mechanism"
        if any(marker in question_text for marker in ("volume", "объем", "объём")) and not any(marker in text for marker in ("volume", "count", "record count", "объем", "объём")):
            return False, "ignored_hypothesis_mechanism"
    if metric and _looks_like_wrong_metric_answer(text, metric):
        return False, "wrong_metric"
    if dimension and not time_axis and branch_type not in {"trend_analysis", "temporal_decomposition"} and _looks_like_wrong_dimension_answer(text, dimension):
        return False, "wrong_dimension"
    filler = ("i would keep this follow-up", "the current evidence says the answer should be judged", "next useful step", "the active conclusion is")
    if any(marker in text for marker in filler):
        return False, "workflow_filler"
    system_like = ("this continues the current chart scope", "this is a temporal trend calculation", "i ranked all groups", "i ranked all `")
    if any(marker in text for marker in system_like):
        return False, "system_like_analytical_language"
    return True, ""


def _is_quality_question(text: str) -> bool:
    return any(marker in text for marker in ("quality", "missing", "duplicate", "duplicates", "null", "пропуск", "дублик", "качеств", "повтор"))


def _is_hypothesis_question(text: str) -> bool:
    if _is_hypothesis_followup_question(text):
        return False
    return str(text or "").startswith(("hypothesis:", "hypothesis -")) or "hypothesis:" in str(text or "")


def _is_hypothesis_followup_question(text: str) -> bool:
    return any(
        marker in str(text or "")
        for marker in (
            "what supports",
            "what contradicts",
            "contradict",
            "counterevidence",
            "counter evidence",
            "evidence supports",
            "supports this",
            "поддерж",
            "противореч",
            "опроверг",
        )
    )


def _wrong_response_language(question: str, response_text: str, state: dict[str, Any]) -> bool:
    protected = [
        str(state.get("active_metric") or ""),
        str(state.get("active_dimension") or ""),
        str(state.get("active_time_axis") or ""),
    ]
    target = state.get("active_analytical_target") if isinstance(state.get("active_analytical_target"), dict) else {}
    protected.extend(str(item) for item in target.get("active_entities", []) if str(item).strip())
    body = _strip_code_spans(response_text)
    for term in protected:
        if term:
            body = body.replace(term, " ")
    cyr = sum(1 for char in body if "а" <= char.lower() <= "я" or char.lower() == "ё")
    cyr_tokens = [
        token
        for token in body.split()
        if any("а" <= char.lower() <= "я" or char.lower() == "ё" for char in token)
    ]
    return cyr >= 16 or len(cyr_tokens) >= 4


def _strip_code_spans(text: str) -> str:
    parts = str(text or "").split("`")
    return " ".join(part for idx, part in enumerate(parts) if idx % 2 == 0)


def _looks_like_shallow_hypothesis(text: str) -> bool:
    compact = " ".join(text.split())
    shallow = (
        compact in {"the hypothesis is supported.", "the hypothesis is not supported.", "гипотеза поддерживается.", "гипотеза не поддерживается."}
        or (compact.startswith("the hypothesis is supported") and "confidence" not in compact and "evidence" not in compact)
        or (compact.startswith("гипотеза поддерживается") and "уверенность" not in compact and "данн" not in compact)
    )
    return shallow


def _looks_like_grouped_country_answer(text: str) -> bool:
    return "`country`" in text and any(marker in text for marker in ("ranked all", "average `sales`", "sales by country", "groups by average"))


def _is_shipping_behavior_question(text: str) -> bool:
    return any(marker in text for marker in ("shipping", "ship ", "ship mode", "delivery", "достав"))


def _looks_like_raw_ship_date_grouping(text: str) -> bool:
    return "`ship date`" in text and any(
        marker in text
        for marker in (
            "ship date contributors",
            "`ship date` contributors",
            "groups by `ship date`",
            "over time by `ship date`",
            "ranked all `ship date`",
            "compared `sales` over time by `ship date`",
        )
    )


def _is_affected_findings_question(text: str) -> bool:
    return any(marker in text for marker in ("finding", "findings", "вывод")) and any(marker in text for marker in ("unreliable", "reliable", "weaken", "ненад"))


def _looks_generic_affected_findings(text: str) -> bool:
    generic = (
        "duplicates can inflate counts" in text
        or "missingness can bias segment comparisons" in text
        or "quality issues matter because" in text
    )
    concrete = any(marker in text for marker in ("total `", "estimated", "inflation", "ranking", "average", "median", "record count"))
    return generic and not concrete


def _asks_transformation_change(text: str) -> bool:
    return any(marker in text for marker in ("what changed", "changed after", "after filtering", "became unreliable", "что измен"))


def _looks_like_adjusted_leaders_only(text: str) -> bool:
    has_leaders = "adjusted ranking" in text or "leaders for" in text or "adjusted leaders" in text
    lacks_comparison = not any(marker in text for marker in ("who dropped", "who stayed", "raw leaders", "rank #", "weaker conclusions", "stronger conclusions"))
    return has_leaders and lacks_comparison


def _contradicts_active_transformation(text: str, state: dict[str, Any]) -> bool:
    adjusted = state.get("active_adjusted_ranking")
    if not isinstance(adjusted, list) or not adjusted:
        return False
    dimension = str(state.get("active_dimension") or "")
    adjusted_names = [
        str(row.get(dimension) or "")
        for row in adjusted[:5]
        if isinstance(row, dict) and str(row.get(dimension) or "").strip()
    ]
    if not adjusted_names:
        return False
    mentions_any_adjusted = any(name and name.lower() in text for name in adjusted_names)
    raw_fragile_names = ("jamestown", "cheyenne", "bellingham")
    mentions_raw_fragile = any(name in text for name in raw_fragile_names)
    return mentions_raw_fragile and not mentions_any_adjusted


def _post_transformation_leaders(
    question: str,
    metric: str,
    dimension: str,
    latest: dict[str, Any],
) -> ConversationResponse:
    chart = latest.get("chart") or {}
    table = latest.get("table") or {}
    rows = _artifact_rows(chart) or _artifact_rows(table)
    if not rows:
        return ConversationResponse(
            text=f"The adjusted `{metric}` by `{dimension}` chart does not include displayed leader values yet, so I need the chart data before explaining who remains ahead.",
            response_kind="clarification_needed",
        )
    value_key = _first_existing_key(rows[0], ("adjusted_mean", "mean", "median", "total"))
    count_key = _first_existing_key(rows[0], ("count", "adjusted_count", "record_count"))
    leaders = rows[:5]
    leader_text = ", ".join(
        f"`{row.get(dimension)}` ({_metric_label(value_key)} {float(row.get(value_key) or 0):.2f}, n={int(float(row.get(count_key) or 0)) if count_key else 'n/a'})"
        for row in leaders
    )
    sparse = [
        str(row.get(dimension))
        for row in leaders
        if count_key and float(row.get(count_key) or 0) <= 2
    ]
    text = (
        f"After filtering, the strongest `{dimension}` groups by `{metric}` are {leader_text}. "
    )
    if sparse:
        text += "The top remains fragile: " + ", ".join(f"`{item}`" for item in sparse[:4]) + " have very small samples. "
    text += "The key point is that the apparent leaders are the groups that still hold up after the outlier and sample-size adjustment."
    return ConversationResponse(
        text=text,
        findings=[text],
        updated_state={
            "active_metric": metric,
            "active_dimension": dimension,
            "active_transformation": latest.get("analysis_type", ""),
            "last_successful_analysis_type": "post_transformation_followup",
            "last_successful_answer_summary": text[:500],
        },
        response_kind="post_transformation_followup",
    )


def _post_transformation_impact(
    question: str,
    metric: str,
    dimension: str,
    latest: dict[str, Any],
) -> ConversationResponse:
    impact = latest.get("impact") if isinstance(latest.get("impact"), dict) else {}
    comparison_rows = impact.get("comparison_rows") if isinstance(impact.get("comparison_rows"), list) else []
    if not comparison_rows:
        return _post_transformation_leaders(question, metric, dimension, latest)
    dropped = _impact_bucket(impact, ("rank_shift", "dropped")) or [
        row for row in comparison_rows if float(row.get("rank_change") or 0) >= 5 or float(row.get("adjusted_rank") or 0) > 20
    ][:4]
    stable = _impact_bucket(impact, ("rank_shift", "stable")) or [
        row for row in comparison_rows if float(row.get("original_rank") or 0) <= 10 and float(row.get("adjusted_rank") or 0) <= 10 and abs(float(row.get("rank_change") or 0)) <= 3
    ][:4]
    fragile = _impact_bucket(impact, ("outlier_dependence", "fragile_leaders"))
    adjusted = impact.get("adjusted_ranking") if isinstance(impact.get("adjusted_ranking"), list) else _artifact_rows(latest.get("chart") or latest.get("table") or {})
    parts = [
        f"Filtering changed the `{metric}` by `{dimension}` story from a raw average ranking into a robustness comparison.",
    ]
    if dropped:
        parts.append(f"Who dropped: {_shift_phrase(dropped[:4], dimension)}.")
    if stable:
        parts.append(f"Who stayed stable: {_shift_phrase(stable[:4], dimension)}.")
    if adjusted:
        parts.append(f"Adjusted leaders now are {_leader_phrase(adjusted[:5], dimension)}.")
    if fragile:
        parts.append(f"Remaining weak spots: {_leader_phrase(fragile[:4], dimension)} still have tiny sample sizes.")
    confidence = str(impact.get("confidence_change") or "")
    if confidence:
        parts.append(confidence)
    weaker = impact.get("weaker_conclusions") if isinstance(impact.get("weaker_conclusions"), list) else []
    stronger = impact.get("stronger_conclusions") if isinstance(impact.get("stronger_conclusions"), list) else []
    if weaker:
        parts.append("Weaker conclusions: " + " ".join(str(item) for item in weaker[:2]))
    if stronger:
        parts.append("Stronger conclusions: " + " ".join(str(item) for item in stronger[:2]))
    text = " ".join(parts)
    return ConversationResponse(
        text=text,
        findings=[text],
        updated_state={
            "active_metric": metric,
            "active_dimension": dimension,
            "active_transformation": latest.get("analysis_type", ""),
            "last_successful_analysis_type": "transformation_impact",
            "last_successful_answer_summary": text[:500],
        },
        response_kind="transformation_impact",
    )


def _hypothesis_response(
    question: str,
    metric: str,
    dimension: str,
    latest: dict[str, Any],
) -> ConversationResponse | None:
    if not metric or not dimension:
        return None
    adjusted = bool(latest)
    language = ResponseLanguagePolicy.from_message(question, protected_terms=[metric, dimension])
    if language.is_russian:
        base = (
            f"Гипотеза: различия `{metric}` между группами `{dimension}` частично объясняются не устойчивой силой самих групп, "
            "а сочетанием extreme records, размера выборки и объема наблюдений."
        )
        if adjusted:
            base += (
                " После удаления extreme records raw-лидеры меняются или проседают, поэтому исходный рейтинг, вероятно, был outlier-driven; "
                "надежнее считать сильными только группы, которые остаются высоко в adjusted/median ranking и имеют достаточный n."
            )
        else:
            base += " Проверять ее нужно сравнением raw average, median, adjusted average и group count."
    else:
        base = (
            f"Hypothesis: differences in `{metric}` across `{dimension}` may be explained by extreme records, sample size, and record volume rather than stable group strength."
        )
        if adjusted:
            base += " After removing extreme records, raw leaders shift or drop, so the original ranking may be outlier-driven; stronger leaders should remain high in adjusted or median rankings with sufficient n."
        else:
            base += " Validate it by comparing raw average, median, adjusted average, and group count."
    return ConversationResponse(
        text=base,
        findings=[base],
        updated_state={
            "active_metric": metric,
            "active_dimension": dimension,
            "active_hypothesis": base,
            "last_successful_analysis_type": "hypothesis",
            "last_successful_answer_summary": base[:500],
        },
        response_kind="hypothesis",
    )


def _validation_response(
    question: str,
    metric: str,
    dimension: str,
    latest: dict[str, Any],
    state: dict[str, Any],
) -> ConversationResponse | None:
    if not metric or not dimension:
        return None
    hypothesis = str(state.get("active_hypothesis") or "").strip()
    language = ResponseLanguagePolicy.from_message(question, protected_terms=[metric, dimension])
    if language.is_russian:
        adjusted_note = "уже есть adjusted ranking после трансформации; " if latest else ""
        text = (
            f"Проверять это нужно как одну цепочку по `{metric}` by `{dimension}`: "
            f"1. сравнить raw average ranking с median ranking; "
            f"2. {adjusted_note}сравнить raw ranking с ranking после удаления extreme records; "
            "3. отфильтровать группы с маленьким n и посмотреть, кто остается наверху; "
            "4. отдельно проверить total metric vs record count, чтобы отделить эффект объема от average value; "
            "5. проверить rank stability: какие группы сохраняют позицию между raw, median и adjusted ranking. "
            "Гипотеза подтверждается, если raw-лидеры проседают после фильтрации, а устойчивые лидеры имеют достаточный размер выборки."
        )
        if hypothesis:
            text = f"Для текущей гипотезы: {hypothesis} {text}"
    else:
        adjusted_note = "there is already an adjusted ranking after transformation; " if latest else ""
        text = (
            f"Validate this as one chain for `{metric}` by `{dimension}`: "
            "1. compare raw average ranking with median ranking; "
            f"2. {adjusted_note}compare raw ranking with ranking after removing extreme records; "
            "3. filter small-n groups and see who remains near the top; "
            "4. compare total metric against record count to separate volume from average value; "
            "5. check rank stability between raw, median, and adjusted rankings. "
            "The hypothesis is supported if raw leaders drop after filtering while stable leaders keep enough sample size."
        )
        if hypothesis:
            text = f"For the current hypothesis: {hypothesis} {text}"
    return ConversationResponse(
        text=text,
        findings=[text],
        updated_state={
            "active_metric": metric,
            "active_dimension": dimension,
            "active_validation_question": question,
            "last_successful_analysis_type": "validation_plan",
            "last_successful_answer_summary": text[:500],
        },
        response_kind="validation_plan",
    )


def _latest_transformation_artifacts(artifacts: list[Any]) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    for artifact in reversed(artifacts):
        metadata = _metadata(artifact)
        nested = metadata.get("metadata") if isinstance(metadata.get("metadata"), dict) else {}
        analysis_type = str(metadata.get("analysis_type") or nested.get("analysis_type") or "").strip()
        if not analysis_type:
            continue
        if analysis_type in {"remove_outliers", "median_instead_of_mean", "normalize_by_volume", "exclude_sparse_groups", "stability_check"}:
            latest.setdefault("analysis_type", analysis_type)
            artifact_type = _artifact_type(artifact)
            if artifact_type == "chart" and "chart" not in latest:
                latest["chart"] = artifact
            if artifact_type == "table" and "table" not in latest:
                latest["table"] = artifact
    return latest


def _latest_transformation_from_state(state: dict[str, Any]) -> dict[str, Any]:
    payload = state.get("active_transformation_result")
    transformed = transformation_state_from_payload(payload)
    if not transformed or not transformed.adjusted_ranking:
        adjusted = state.get("active_adjusted_ranking")
        if not isinstance(adjusted, list) or not adjusted:
            return {}
        transformed = transformation_state_from_payload(
            {
                "metric": state.get("active_metric") or "",
                "dimension": state.get("active_dimension") or "",
                "transformation_type": state.get("active_transformation") or "transformation",
                "adjusted_ranking": adjusted,
                "ranking_scope": state.get("active_ranking_scope") or "",
            }
        )
    if not transformed or not transformed.adjusted_ranking:
        return {}
    rows = transformed.adjusted_ranking
    return {
        "analysis_type": transformed.transformation_type,
        "metric": transformed.metric,
        "dimension": transformed.dimension,
        "ranking_scope": transformed.ranking_scope,
        "impact": transformed.to_payload(),
        "chart": {"content": {"rows": rows}},
        "table": {"content": transformed.comparison_rows or rows},
    }


def _impact_bucket(impact: dict[str, Any], path: tuple[str, str]) -> list[dict[str, Any]]:
    parent = impact.get(path[0])
    if not isinstance(parent, dict):
        return []
    rows = parent.get(path[1])
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _shift_phrase(rows: list[dict[str, Any]], dimension: str) -> str:
    parts = []
    for row in rows:
        value = row.get(dimension)
        parts.append(
            f"`{value}` rank #{int(float(row.get('original_rank') or 0))} -> #{int(float(row.get('adjusted_rank') or 0))} "
            f"(avg {float(row.get('original_mean') or 0):.2f} -> {float(row.get('adjusted_mean') or 0):.2f})"
        )
    return ", ".join(parts) if parts else "none"


def _leader_phrase(rows: list[dict[str, Any]], dimension: str) -> str:
    parts = []
    for row in rows:
        value = row.get(dimension)
        mean = row.get("mean", row.get("adjusted_mean", 0))
        count = row.get("count", row.get("adjusted_count", 0))
        parts.append(f"`{value}` (avg {float(mean or 0):.2f}, n={int(float(count or 0))})")
    return ", ".join(parts) if parts else "none"


def _artifact_rows(artifact: Any) -> list[dict[str, Any]]:
    content = artifact.get("content") if isinstance(artifact, dict) else getattr(artifact, "content", None)
    if isinstance(content, dict) and isinstance(content.get("rows"), list):
        return [row for row in content["rows"] if isinstance(row, dict)]
    if isinstance(content, list):
        return [row for row in content if isinstance(row, dict)]
    return []


def _content(artifact: Any) -> dict[str, Any]:
    content = artifact.get("content") if isinstance(artifact, dict) else getattr(artifact, "content", None)
    return content if isinstance(content, dict) else {}


def _metadata(artifact: Any) -> dict[str, Any]:
    metadata = artifact.get("metadata") if isinstance(artifact, dict) else getattr(artifact, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def _title(artifact: Any) -> str:
    return str(artifact.get("title") if isinstance(artifact, dict) else getattr(artifact, "title", "") or "").strip()


def _artifact_type(artifact: Any) -> str:
    if isinstance(artifact, dict):
        return str(artifact.get("artifact_type") or "")
    return str(getattr(getattr(artifact, "artifact_type", None), "value", getattr(artifact, "artifact_type", "")))


def _is_chart_explanation_question(question: str) -> bool:
    text = _normalize(question)
    return any(marker in text for marker in ("explain the chart", "explain the adjusted chart", "explain adjusted chart", "explain this chart", "explain sales by", "explain this", "main takeaway", "adjusted chart", "объясни график", "объясни диаграм", "что значит график"))


def _resolve_chart_artifact(question: str, artifacts: list[Any], artifact_id: str = "") -> Any | None:
    text = _normalize(question)
    charts = [artifact for artifact in artifacts if _artifact_type(artifact) == "chart"]
    if not charts:
        return None
    if artifact_id:
        for artifact in charts:
            if str(artifact.get("artifact_id") if isinstance(artifact, dict) else getattr(artifact, "artifact_id", "")) == artifact_id:
                return artifact
        return None
    if any(marker in text for marker in ("adjusted", "filtered", "after removing", "after filtering", "скоррект", "после фильтр", "без выброс")):
        for artifact in reversed(charts):
            metadata = _metadata(artifact)
            content = _content(artifact)
            analysis_type = str(metadata.get("analysis_type") or metadata.get("transformation_type") or content.get("transformation_type") or "").strip()
            if analysis_type:
                return artifact
    for artifact in reversed(charts):
        title = _normalize(_title(artifact))
        metadata = _metadata(artifact)
        content = _content(artifact)
        metric = _normalize(str(metadata.get("metric") or content.get("metric") or ""))
        dimension = _normalize(str(metadata.get("dimension") or content.get("dimension") or content.get("x") or ""))
        if title and title in text:
            return artifact
        if metric and dimension and metric in text and dimension in text:
            return artifact
    return charts[-1]


def _artifact_value_key(row: dict[str, Any], aggregation: str) -> str:
    normalized = _normalize(aggregation)
    if normalized in row:
        return normalized
    if normalized == "sum" and "total" in row:
        return "total"
    if normalized == "total" and "sum" in row:
        return "sum"
    for key in ("total", "sum", "mean", "median", "value", "count"):
        if key in row:
            return key
    return next(iter(row.keys()))


def _aggregation_label(aggregation: str) -> str:
    normalized = _normalize(aggregation)
    if normalized in {"sum", "total"}:
        return "total"
    if normalized == "mean":
        return "average"
    if normalized == "median":
        return "median"
    if normalized == "count":
        return "count of"
    return normalized or "charted"


def _filter_phrase(filters: Any) -> str:
    if not isinstance(filters, list) or not filters:
        return ""
    parts = []
    for item in filters:
        if isinstance(item, dict) and item.get("column") and item.get("value") is not None:
            parts.append(f"`{item['column']}` = `{item['value']}`")
    return " for " + ", ".join(parts) if parts else ""


def _looks_like_wrong_metric_answer(text: str, active_metric: str) -> bool:
    forbidden = ("postal code", "postcode", "zip", "row id", "customer id")
    return _normalize(active_metric) not in text and any(marker in text for marker in forbidden)


def _looks_like_wrong_dimension_answer(text: str, active_dimension: str) -> bool:
    if _normalize(active_dimension) in text:
        return False
    return "ship mode" in text or "`ship mode`" in text


def _first_existing_key(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in row:
            return key
    return keys[0]


def _metric_label(key: str) -> str:
    return "avg" if "mean" in key else key.replace("_", " ")


def _topic(metric: str, dimension: str) -> str:
    if metric and dimension:
        return f"`{metric}` by `{dimension}`"
    return f"`{metric or dimension}`"


def _normalize(value: str) -> str:
    return " ".join(str(value or "").replace("_", " ").replace("-", " ").lower().split())
