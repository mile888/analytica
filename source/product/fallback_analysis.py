from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

import pandas as pd

from source.dataframe import parse_glued_dataframe
from source.product.analytical_branches import (
    choose_behavior_dimension,
    choose_decomposition_dimension,
    route_branch_intent,
    seasonality_response,
    strongest_growth_response,
    temporal_anomalies_response,
    temporal_behavior_response,
    temporal_decomposition_response,
    trend_summary_response,
)
from source.product.affected_findings import AffectedFindingsAnalyzer, is_affected_findings_question
from source.product.analytical_graph import BranchTransitionSynthesizer
from source.product.branch_workspace import BranchWorkspaceManager
from source.product.execution_planner import (
    AuthoritativeExecutionPlanner,
    AuthoritativeQueryPlan,
    FilterConstraintValidator,
    NonAnalyticalUtteranceClassifier,
    TemporalSanityValidator,
)
from source.product.execution_context import (
    ExecutionContextUnavailableError,
    execution_context_failure_output,
    execution_required_for_question,
    execution_unavailable_in_context,
)
from source.product.direct_query_executor import DirectQueryExecutor
from source.product.semantic_layer import (
    build_investigation_thread_state,
    build_investigation_focus,
    build_semantic_dataset_profile,
    match_entity_or_value,
    select_dimension_column as select_semantic_dimension_column,
    select_metric_column as select_semantic_metric_column,
    select_timestamp_column as select_semantic_timestamp_column,
    semantic_selection_text,
    thread_selection_text,
)
from source.product.conversation_engine import answer_from_conversation_state
from source.product.evidence_resolution import (
    build_active_target,
    evidence_response_text,
    is_evidence_followup,
    resolve_evidence_subject,
    target_quality_next_steps,
)
from source.product.hypothesis_reasoning import HypothesisEngine, is_hypothesis_question
from source.product.language_policy import ResponseLanguagePolicy, average_chart_title
from source.product.transformation_engine import (
    is_transformation_question,
    run_transformation_analysis,
    transformation_clarification_response,
)
from source.product.fallbacks.semantic_resolution import (
    DELIVERY_DATE_MARKERS,
    ORDER_DATE_MARKERS,
    SHIPPING_METHOD_MARKERS,
    _column_name_matches,
    _contains_any,
    _dimension_explicitly_requested,
    _dimension_semantic_groups,
    _explicit_dimension_column,
    _explicit_metric_column,
    _has_correlation_candidate,
    _looks_identifier_like,
    _mentioned_columns,
    _missing_explicit_entity,
    _missing_explicit_metric_request,
    _normalize,
    _select_dimension_column,
    _select_metric_column,
    _select_timestamp_column,
    _semantic_column_by_markers,
    _semantic_column_match,
    _valid_column,
    _is_meaningful_dimension,
    resolve_categorical_value,
)
from source.product.fallbacks.artifact_builders import build_histogram_artifact
from source.product.fallbacks.metric_analysis import (
    _correlation_strength,
    _correlation_response,
    _outlier_response,
    _single_metric_response,
    _trend_response,
)
from source.product.fallbacks.narration import _output, _timeline
from source.product.fallbacks.quality_analysis import (
    _data_quality_response,
    _duplicate_impact_response,
    _quality_issue_impact_response,
)
from source.product.fallbacks.shipping_analysis import build_delivery_delay_artifacts, compute_delivery_delay_analysis
from source.product.fallbacks.validation import (
    _is_analytical_follow_up,
    _is_business_questions_request,
    _is_chart_question,
    _is_composition_question,
    _is_context_reset_question,
    _is_contextual_follow_up,
    _is_correlation_question,
    _is_dependency_question,
    _is_distribution_question,
    _is_duplicate_impact_followup,
    _is_duplicate_impact_question,
    _is_duplicate_question,
    _is_findings_evidence_question,
    _is_hypothesis_validation_question,
    _is_important_fields_request,
    _is_investigation_reset_question,
    _is_next_check_question,
    _is_outlier_question,
    _is_overview_question,
    _is_quality_issue_impact_question,
    _is_quality_question,
    _is_ranking_or_group_comparison_question,
    _is_trend_question,
    _is_volume_relationship_follow_up,
    _metric_is_explicit_enough,
    _should_continue_from_context,
)


AUTHORITATIVE_FOLLOW_UP_TYPES = {
    "follow_up",
    "anomaly",
    "validation",
    "causal_hypothesis",
    "comparison",
    "explanation",
    "analytical_transformation",
}


def deterministic_context_fallback(question: str, data_context: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a useful answer from product data-source context when raw data is unavailable."""

    contexts = (data_context or {}).get("data_source_usage_contexts") or []
    if not isinstance(contexts, list) or not contexts:
        return None

    context = contexts[0]
    if not isinstance(context, dict):
        return None

    conversation_context = (data_context or {}).get("conversation_context") or (data_context or {}).get("conversation") or {}
    if execution_required_for_question(question, data_context=data_context) and execution_unavailable_in_context(data_context):
        return execution_context_failure_output(
            question,
            ExecutionContextUnavailableError(
                "Executable dataset rows are unavailable. The saved schema/profile can be used for metadata reasoning only.",
                data_source_id=context.get("data_source_id"),
                reason="profile_only_mode",
            ),
        )
    if _is_findings_evidence_question(question):
        return _findings_evidence_context_response(question, conversation_context, context)
    if _is_analytical_follow_up(question):
        return _profile_based_analytical_response(question, conversation_context, context)
    if _should_continue_from_context(question, conversation_context):
        return _continuity_context_response(question, conversation_context, context)

    name = str(context.get("name") or "selected data source")
    schema = context.get("schema_summary") if isinstance(context.get("schema_summary"), dict) else {}
    row_count = int(schema.get("row_count") or schema.get("rows") or 0)
    column_count = int(schema.get("column_count") or schema.get("columns") or 0)
    columns = _column_summaries_from_context(context)
    metric_columns = [item["name"] for item in columns if item["role"] == "metric"]
    dimension_columns = [item["name"] for item in columns if item["role"] == "dimension"]
    timestamp_columns = [item["name"] for item in columns if item["role"] == "timestamp"]
    caveats = [str(item) for item in context.get("caveats", []) if str(item).strip()]

    summary = _dataset_overview_summary(
        name=name,
        row_count=row_count,
        column_count=column_count,
        metrics=metric_columns,
        dimensions=dimension_columns,
        timestamps=timestamp_columns,
        caveats=caveats,
    )
    findings: list[str] = []

    evidence = [
        f"Data source profile for `{name}`.",
        f"Column roles inferred from {len(columns)} column summaries.",
    ]
    limitations = [
        "This answer is based on the saved dataset profile, not a fresh scan of the raw file.",
    ]
    if caveats:
        limitations.extend(caveats[:4])
    next_steps = _analytical_directions(metric_columns, dimension_columns, timestamp_columns)[:5] or [
        "Ask a focused follow-up about a metric, segment, time trend, anomaly, or evidence strength.",
    ]

    profile_table = pd.DataFrame(columns[:12])
    result_preview = profile_table.to_string(index=False) if not profile_table.empty else ""
    timeline = _timeline("data_source_context_fallback", rows=row_count, columns=column_count)
    return _output(
        question=question,
        summary=summary,
        findings=findings,
        evidence=evidence,
        limitations=limitations,
        next_steps=next_steps,
        code="",
        result_preview=result_preview,
        timeline=timeline,
        artifacts=[
            {
                "artifact_type": "table",
                "title": f"{name} column summary",
                "content": profile_table.to_dict(orient="records") if not profile_table.empty else [],
                "visibility": "technical",
                "pinned": True,
                "metadata": {"source": "data_source_usage_context", "analysis_type": "overview"},
            }
        ],
        trace_metadata={
            "fallback": "data_source_context",
            "data_source_name": name,
            "analysis_type": "overview",
            "suppress_key_findings": True,
        },
    )


def deterministic_investigation_fallback(
    question: str,
    df: Any,
    *,
    data_context: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Generic product fallback for tabular investigations.

    This keeps the Investigation Workspace useful when the LLM returns prose
    without invoking analytics tools. It intentionally avoids dataset-specific
    filenames, business domains, and column names.
    """

    if not isinstance(df, pd.DataFrame) or df.empty:
        if execution_required_for_question(question, data_context=data_context) and execution_unavailable_in_context(data_context):
            return execution_context_failure_output(
                question,
                ExecutionContextUnavailableError(
                    "Executable dataset rows are unavailable or empty, so this analytical query cannot be executed.",
                    reason="dataframe_unavailable",
                ),
            )
        return None

    conversation_context = (data_context or {}).get("conversation_context") if isinstance(data_context, dict) else {}
    df = _dataframe_with_derived_columns(df, conversation_context if isinstance(conversation_context, dict) else {})

    if _is_context_reset_question(question):
        return _dataframe_exploration_response(question, df)
    if _is_business_questions_request(question):
        return _business_questions_response(question, df)
    if _is_important_fields_request(question):
        return _important_fields_response(question, df)
    if _is_overview_question(question) and _is_dependency_question(question):
        return _dataframe_overview_with_dependencies_response(question, df)
    if _is_overview_question(question):
        return _dataframe_exploration_response(question, df)
    if _is_ingestion_diagnostic_question(question):
        return _ingestion_diagnostic_response(question, df)

    early_state = conversation_context.get("conversation_state") if isinstance(conversation_context, dict) and isinstance(conversation_context.get("conversation_state"), dict) else {}
    if _is_explicit_data_quality_scan_question(question):
        explicit_metric = _explicit_metric_column(df, question)
        if _is_duplicate_impact_question(question) or _is_duplicate_impact_followup(question, early_state):
            return _duplicate_impact_response(question, df, explicit_metric, None)
        if _is_quality_issue_impact_question(question):
            return _quality_issue_impact_response(question, df, explicit_metric, None)
        return _data_quality_response(question, df)
    dataframe_operation = _dataframe_operation_response(question, df, conversation_context if isinstance(conversation_context, dict) else {})
    if dataframe_operation:
        return dataframe_operation
    if NonAnalyticalUtteranceClassifier.is_non_analytical(question):
        return _output(
            question=question,
            summary="That sounds like a comment about the agent behavior, not a data-analysis request. Ask a concrete analytical question or choose a branch to continue.",
            findings=[],
            evidence=[],
            limitations=[],
            next_steps=[],
            code="",
            result_preview="",
            timeline=[{"tool": "non_analytical_utterance_classifier", "status": "ok"}],
            artifacts=[],
            trace_metadata={"fallback": "non_analytical_utterance", "analysis_type": "non_analytical", "suppress_key_findings": True},
        )
    distribution_compare = _distribution_comparison_followup(question, df, conversation_context if isinstance(conversation_context, dict) else {})
    if distribution_compare:
        return distribution_compare
    bins_followup = _bins_followup_output(question, df, conversation_context if isinstance(conversation_context, dict) else {})
    if bins_followup:
        return bins_followup
    if _should_route_to_transformed_state(question, conversation_context if isinstance(conversation_context, dict) else {}):
        state_answer = answer_from_conversation_state(
            question=question,
            conversation_context=conversation_context if isinstance(conversation_context, dict) else {},
            recent_artifacts=(conversation_context.get("recent_artifacts") if isinstance(conversation_context, dict) else []) or [],
        )
        if state_answer:
            return _output(
                question=question,
                summary=state_answer.text,
                findings=state_answer.findings,
                evidence=["Answered from the active transformed analytical artifact."],
                limitations=[],
                next_steps=[],
                code="",
                result_preview="",
                timeline=[{"tool": "conversation_engine", "status": "ok", "response_kind": state_answer.response_kind}],
                artifacts=state_answer.artifacts,
                trace_metadata={
                    "fallback": "conversation_engine",
                    "analysis_type": state_answer.response_kind,
                    **state_answer.updated_state,
                },
            )
    authoritative = _authoritative_plan_output(question, df, conversation_context if isinstance(conversation_context, dict) else {})
    if authoritative:
        return authoritative
    missing_metric = _missing_explicit_metric_request(question, [str(col) for col in df.columns])
    if missing_metric and not (is_hypothesis_question(question) or _is_hypothesis_validation_question(question)):
        semantic = build_semantic_dataset_profile(df=df)
        ranked_metrics = [item["name"] for item in _rank_metric_candidates(df, semantic)[:6]]
        return _missing_metric_response(question, missing_metric, ranked_metrics, source_name="the dataset")
    thread_state = build_investigation_thread_state(
        conversation_context if isinstance(conversation_context, dict) else {},
        question=question,
    )
    focus = build_investigation_focus(conversation_context if isinstance(conversation_context, dict) else {}, question=question)
    semantic_profile = build_semantic_dataset_profile(df=df)
    state = conversation_context.get("conversation_state") if isinstance(conversation_context, dict) and isinstance(conversation_context.get("conversation_state"), dict) else {}
    explicit_quality_branch = _is_quality_question(question) or _is_duplicate_impact_followup(question, state)
    explicit_hypothesis_branch = is_hypothesis_question(question) or _is_hypothesis_validation_question(question)
    affected = _affected_findings_output(question, df, state, conversation_context if isinstance(conversation_context, dict) else {})
    if affected:
        return affected
    active_evidence = _active_target_evidence_output(question, state, conversation_context if isinstance(conversation_context, dict) else {})
    if active_evidence:
        return active_evidence
    quality_next = _quality_branch_next_steps_output(question, state)
    if quality_next:
        return quality_next
    selection_text = _selection_text_for_question(question, data_context)
    semantic_text = thread_selection_text(semantic_selection_text(selection_text, focus), thread_state)
    metric_col, dimension_col, focus_authority = _resolve_authoritative_focus(
        df=df,
        question=question,
        conversation_context=conversation_context if isinstance(conversation_context, dict) else {},
        semantic_profile=semantic_profile,
    )
    if explicit_quality_branch:
        metric_for_quality = _explicit_metric_column(df, question) or (metric_col if _metric_is_explicit_enough(question, metric_col) else None)
        if _is_duplicate_impact_question(question) or _is_duplicate_impact_followup(question, state):
            return _duplicate_impact_response(question, df, metric_for_quality, None)
        if _is_quality_issue_impact_question(question):
            return _quality_issue_impact_response(question, df, metric_for_quality, None)
        return _data_quality_response(question, df)
    if explicit_hypothesis_branch:
        hypothesis = HypothesisEngine.validate(
            question=question,
            dataframe=df,
            branch_state=state,
            semantic_profile=semantic_profile,
        )
        if hypothesis:
            return hypothesis
    value_match = match_entity_or_value(question, semantic_profile, df=df)
    if value_match and not dimension_col:
        dimension_col = value_match.matched_column
    if is_transformation_question(question) and (not metric_col or not dimension_col):
        return transformation_clarification_response(question)
    state_answer = answer_from_conversation_state(
        question=question,
        conversation_context=conversation_context if isinstance(conversation_context, dict) else {},
        recent_artifacts=(conversation_context.get("recent_artifacts") if isinstance(conversation_context, dict) else []) or [],
    )
    if state_answer:
        return _output(
            question=question,
            summary=state_answer.text,
            findings=state_answer.findings,
            evidence=["Answered from active investigation state and latest analytical artifacts."],
            limitations=["This answer uses the latest saved analytical result as the active context."],
            next_steps=[],
            code="",
            result_preview="",
            timeline=[{"tool": "conversation_engine", "status": "ok", "response_kind": state_answer.response_kind}],
            artifacts=state_answer.artifacts,
            trace_metadata={
                "fallback": "conversation_engine",
                "analysis_type": state_answer.response_kind,
                **state_answer.updated_state,
            },
        )
    if not metric_col:
        metric_col = select_semantic_metric_column(
            semantic_profile,
            semantic_text,
            explicit_text=question,
            focus=focus,
        ) or _select_metric_column(df, selection_text, explicit_question=question)
    if not dimension_col:
        dimension_col = select_semantic_dimension_column(
            semantic_profile,
            semantic_text,
            exclude={metric_col} if metric_col else set(),
            explicit_text=question,
            focus=focus,
        ) or _select_dimension_column(
            df,
            selection_text,
            exclude={metric_col} if metric_col else set(),
            explicit_question=question,
        )
    timestamp_col = select_semantic_timestamp_column(
        semantic_profile,
        semantic_text,
        explicit_text=question,
        focus=focus,
    ) or _select_timestamp_column(df, selection_text)
    chart = conversation_context.get("latest_chart_context") if isinstance(conversation_context, dict) and isinstance(conversation_context.get("latest_chart_context"), dict) else {}
    timestamp_col = _valid_column(df, state.get("active_time_axis")) or _valid_column(df, chart.get("time_axis")) or timestamp_col
    branch_intent = route_branch_intent(question, state)
    if metric_col and timestamp_col and branch_intent in {
        "trend_summary",
        "strongest_growth_periods",
        "temporal_decomposition",
        "seasonality_check",
        "temporal_anomalies",
        "temporal_behavior_decomposition",
    }:
        if branch_intent == "trend_summary":
            return trend_summary_response(question, df, metric_col, timestamp_col)
        if branch_intent == "strongest_growth_periods":
            return strongest_growth_response(question, df, metric_col, timestamp_col)
        if branch_intent == "temporal_decomposition":
            decomposition_col = choose_decomposition_dimension(df, metric_col, timestamp_col, dimension_col)
            if decomposition_col:
                return temporal_decomposition_response(question, df, metric_col, timestamp_col, decomposition_col)
        if branch_intent == "seasonality_check":
            return seasonality_response(question, df, metric_col, timestamp_col)
        if branch_intent == "temporal_anomalies":
            return temporal_anomalies_response(question, df, metric_col, timestamp_col)
        if branch_intent == "temporal_behavior_decomposition":
            behavior_col = choose_behavior_dimension(df, metric_col, timestamp_col, question)
            if behavior_col:
                return temporal_behavior_response(question, df, metric_col, timestamp_col, behavior_col)
    if _is_quality_question(question):
        if _is_duplicate_impact_question(question):
            return _duplicate_impact_response(question, df, metric_col, None)
        if _is_quality_issue_impact_question(question):
            return _quality_issue_impact_response(question, df, metric_col, dimension_col)
        return _data_quality_response(question, df)
    if not metric_col:
        return _metric_coverage_response(question, df)
    missing_entity = _missing_explicit_entity(question, df)
    if missing_entity:
        return _missing_entity_response(question, df, metric_col, missing_entity)
    if is_transformation_question(question):
        transformed = run_transformation_analysis(
            question=question,
            df=df,
            metric_col=metric_col,
            dimension_col=dimension_col or "",
        )
        if transformed:
            return transformed
    if _is_volume_relationship_follow_up(question) and dimension_col:
        return _group_count_relationship_response(question, df, metric_col, dimension_col)
    if _is_outlier_question(question):
        if dimension_col:
            return _group_unusual_values_response(question, df, metric_col, dimension_col)
        return _outlier_response(question, df, metric_col)
    if _is_correlation_question(question) and _has_correlation_candidate(df, metric_col) and not (
        dimension_col and _is_ranking_or_group_comparison_question(question)
    ):
        return _correlation_response(question, df, metric_col)
    if _is_trend_question(question) and timestamp_col:
        return _trend_response(question, df, metric_col, timestamp_col)
    if _is_distribution_question(question):
        return _single_metric_response(question, df, metric_col, chart_requested=_is_chart_question(question))
    if _is_chart_question(question):
        if _is_correlation_question(question) and _has_correlation_candidate(df, metric_col):
            return _correlation_response(question, df, metric_col)
        if _is_distribution_question(question):
            return _single_metric_response(question, df, metric_col, chart_requested=True)
        if dimension_col and (_is_meaningful_dimension(df, dimension_col) or _dimension_explicitly_requested(question, dimension_col)) and (_is_ranking_or_group_comparison_question(question) or _is_composition_question(question)):
            return _grouped_metric_response(question, df, metric_col, dimension_col, chart_requested=True)
        if _is_trend_question(question) and timestamp_col:
            return _trend_response(question, df, metric_col, timestamp_col, chart_requested=True)
        if dimension_col and (_is_meaningful_dimension(df, dimension_col) or _dimension_explicitly_requested(question, dimension_col)):
            return _grouped_metric_response(question, df, metric_col, dimension_col, chart_requested=True)
        if timestamp_col:
            return _trend_response(question, df, metric_col, timestamp_col, chart_requested=True)
        return _single_metric_response(question, df, metric_col, chart_requested=True)
    if not dimension_col:
        return _single_metric_response(question, df, metric_col)
    if not _is_meaningful_dimension(df, dimension_col) and not _dimension_explicitly_requested(question, dimension_col):
        return _single_metric_response(question, df, metric_col)

    return _grouped_metric_response(question, df, metric_col, dimension_col)


def _should_route_to_transformed_state(question: str, conversation_context: dict[str, Any]) -> bool:
    state = conversation_context.get("conversation_state") if isinstance(conversation_context.get("conversation_state"), dict) else {}
    has_transformation = any(
        state.get(key)
        for key in (
            "active_transformation",
            "active_transformation_result",
            "active_adjusted_ranking",
            "transformation_lineage",
        )
    )
    if not has_transformation:
        return False
    text = _normalize(question)
    transformation_reference = any(
        marker in text
        for marker in (
            "remain",
            "remains",
            "leaders",
            "what changed",
            "changed",
            "after filtering",
            "after removing",
            "adjusted chart",
            "adjusted",
            "compare after filtering",
            "остаются",
            "остались",
            "лидер",
            "после",
            "что измен",
        )
    )
    if not transformation_reference:
        return False
    if any(marker in text for marker in ("finding", "findings", "вывод")):
        return False
    explicit_new_entity = any(
        marker in text
        for marker in (
            "customer",
            "client",
            "product",
            "account",
            "operator",
            "vendor",
            "supplier",
            "employee",
            "user",
            "клиент",
            "продукт",
        )
    )
    return not (explicit_new_entity and not any(marker in text for marker in ("adjusted", "after", "после")))


def _is_ingestion_diagnostic_question(question: str) -> bool:
    text = _normalize(question)
    markers = (
        "delimiter",
        "separator",
        "parsed",
        "loaded correctly",
        "correctly loaded",
        "one column",
        "single column",
        "glued",
        "stuck together",
        "reload",
        "проверь загруж",
        "правильно загруж",
        "разделител",
        "разделитель",
        "перезагруз",
        "одной колон",
        "одна колон",
        "склеен",
        "склеенн",
        "первые 5",
        "первые пять",
    )
    return any(marker in text for marker in markers)


def _is_explicit_data_quality_scan_question(question: str) -> bool:
    text = _normalize(question)
    markers = (
        "quality",
        "missing",
        "duplicates",
        "duplicate",
        "clean",
        "preprocess",
        "null",
        "пропуск",
        "дублик",
        "качеств",
        "очист",
        "предобработ",
    )
    return any(marker in text for marker in markers)


def _ingestion_diagnostic_response(question: str, df: pd.DataFrame) -> dict[str, Any]:
    language = ResponseLanguagePolicy.from_message(question, protected_terms=df.columns)
    parsed = parse_glued_dataframe(df)
    if parsed:
        recovered, delimiter = parsed
        preview = recovered.head(5)
        if language.is_russian:
            summary = (
                f"Датасет выглядит распарсенным неправильно: все данные попали в одну колонку. "
                f"Вероятный разделитель `{_display_delimiter(delimiter)}`; после повторного разбора получается "
                f"{len(recovered.columns)} колонок. Ниже первые 5 строк."
            )
            findings = [
                "Исходный dataframe содержит одну колонку со встроенным разделителем.",
                f"Повторный разбор с разделителем `{_display_delimiter(delimiter)}` восстановил табличную структуру.",
            ]
            limitations = [
                "Диагностика основана на доступных строках текущего dataframe; при сохранении источника нужно перечитать исходный CSV тем же разделителем.",
            ]
            next_steps = [
                f"Перезагрузить CSV с разделителем `{_display_delimiter(delimiter)}`.",
                "После перезагрузки повторить profiling и проверку пропусков.",
            ]
        else:
            summary = (
                f"The dataset appears to be parsed incorrectly: all values landed in one column. "
                f"The likely delimiter is `{_display_delimiter(delimiter)}`; reparsing restores "
                f"{len(recovered.columns)} columns. The first 5 rows are shown below."
            )
            findings = [
                "The current dataframe has one column containing embedded delimiters.",
                f"Reparsing with delimiter `{_display_delimiter(delimiter)}` restores a tabular structure.",
            ]
            limitations = [
                "This diagnostic is based on the currently available dataframe rows; the source CSV should be read again with the detected delimiter.",
            ]
            next_steps = [
                f"Reload the CSV using delimiter `{_display_delimiter(delimiter)}`.",
                "Then rerun profiling and missing-value checks.",
            ]
        rows = preview.to_dict(orient="records")
        return _output(
            question=question,
            summary=summary,
            findings=findings,
            evidence=[f"Current shape: {len(df):,} rows x {len(df.columns):,} column.", f"Recovered shape: {len(recovered):,} rows x {len(recovered.columns):,} columns."],
            limitations=limitations,
            next_steps=next_steps,
            code=f"pd.read_csv(path, sep={delimiter!r})",
            result_preview=preview.to_string(index=False),
            timeline=[{"tool": "csv_ingestion_diagnostic", "status": "delimiter_detected", "delimiter": delimiter, "rows": len(recovered), "columns": len(recovered.columns)}],
            artifacts=[
                {
                    "artifact_type": "table",
                    "title": "Recovered first 5 rows",
                    "content": rows,
                    "visibility": "user",
                    "pinned": True,
                    "metadata": {"analysis_type": "csv_ingestion_diagnostic", "delimiter": delimiter, "row_count": int(len(recovered)), "column_count": int(len(recovered.columns))},
                }
            ],
            trace_metadata={"fallback": "csv_ingestion_diagnostic", "analysis_type": "csv_ingestion_diagnostic", "delimiter": delimiter, "suppress_key_findings": True},
        )

    preview = df.head(5)
    if language.is_russian:
        summary = (
            f"Датасет выглядит загруженным как таблица: {len(df):,} строк и {len(df.columns):,} колонок. "
            "Признаков склейки всех данных в одну колонку не видно. Ниже первые 5 строк."
        )
        findings = ["Колонки уже разделены в текущем dataframe."]
        limitations = ["Это проверка структуры текущего dataframe, а не domain-specific validation значений."]
        next_steps = ["Теперь можно запускать profiling, missing-value check или конкретный анализ по колонкам."]
    else:
        summary = (
            f"The dataset appears to be loaded as a table: {len(df):,} rows and {len(df.columns):,} columns. "
            "I do not see evidence that all data was glued into one column. The first 5 rows are shown below."
        )
        findings = ["The current dataframe columns are already separated."]
        limitations = ["This checks the current dataframe structure, not domain-specific value validity."]
        next_steps = ["You can now run profiling, missing-value checks, or column-specific analysis."]
    return _output(
        question=question,
        summary=summary,
        findings=findings,
        evidence=[f"Current shape: {len(df):,} rows x {len(df.columns):,} columns."],
        limitations=limitations,
        next_steps=next_steps,
        code="df.head(5)",
        result_preview=preview.to_string(index=False),
        timeline=[{"tool": "csv_ingestion_diagnostic", "status": "ok", "rows": len(df), "columns": len(df.columns)}],
        artifacts=[
            {
                "artifact_type": "table",
                "title": "First 5 rows",
                "content": preview.to_dict(orient="records"),
                "visibility": "user",
                "pinned": True,
                "metadata": {"analysis_type": "csv_ingestion_diagnostic", "row_count": int(len(df)), "column_count": int(len(df.columns))},
            }
        ],
        trace_metadata={"fallback": "csv_ingestion_diagnostic", "analysis_type": "csv_ingestion_diagnostic", "suppress_key_findings": True},
    )


def _display_delimiter(delimiter: str) -> str:
    if delimiter == "\t":
        return "\\t"
    return delimiter


def _dataframe_with_derived_columns(df: pd.DataFrame, conversation_context: dict[str, Any]) -> pd.DataFrame:
    state = conversation_context.get("conversation_state") if isinstance(conversation_context.get("conversation_state"), dict) else {}
    derived = state.get("derived_columns")
    if not isinstance(derived, list) or not derived:
        return df
    working = df.copy()
    for item in derived:
        if not isinstance(item, dict):
            continue
        name = str(item.get("column") or item.get("derived_column") or "").strip()
        left = str(item.get("left") or "").strip()
        right = str(item.get("right") or "").strip()
        operator = str(item.get("operator") or "").strip()
        if name and name not in working.columns and left in working.columns and right in working.columns and operator == "*":
            working[name] = pd.to_numeric(working[left], errors="coerce") * pd.to_numeric(working[right], errors="coerce")
    return working


def _dataframe_operation_response(question: str, df: pd.DataFrame, conversation_context: dict[str, Any]) -> dict[str, Any] | None:
    text = _normalize(question)
    if any(marker in text for marker in ("dtype", "dtypes", "data type", "types", "numeric", "convert numeric", "тип", "числов")):
        return _dtype_operation_response(question, df)
    formula = _parse_derived_formula(question, df)
    if formula:
        return _derived_column_response(question, df, formula)
    validation = _parse_formula_validation(question, df)
    if validation:
        return _formula_validation_response(question, df, validation)
    if any(marker in text for marker in ("category inspection", "inspect categories", "unique values", "unique categories", "уникальн")):
        return _category_inspection_response(question, df)
    return None


def _dtype_operation_response(question: str, df: pd.DataFrame) -> dict[str, Any]:
    mentioned = _mentioned_columns([str(col) for col in df.columns], question)
    columns = mentioned or [str(col) for col in df.columns]
    rows = []
    for column in columns[:30]:
        if column not in df.columns:
            continue
        series = df[column]
        converted = pd.to_numeric(series, errors="coerce")
        non_null = int(series.notna().sum())
        converted_non_null = int(converted.notna().sum())
        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "non_null": non_null,
                "numeric_convertible": converted_non_null,
                "conversion_failures": max(0, non_null - converted_non_null),
            }
        )
    target = f" for `{rows[0]['column']}`" if len(rows) == 1 else ""
    summary = f"Dataframe dtype inspection{target}: checked {len(rows)} column{'s' if len(rows) != 1 else ''}. This is a dataframe operation, so no grouped ranking or chart was created."
    return _output(
        question=question,
        summary=summary,
        findings=[summary],
        evidence=["Inspected pandas dtypes and numeric conversion feasibility."],
        limitations=["Type conversion was checked but not applied to the persisted dataset in this step."],
        next_steps=["If conversion failures are acceptable or fixable, run the conversion before metric analysis."],
        code="df.dtypes; pd.to_numeric(series, errors='coerce')",
        result_preview=pd.DataFrame(rows).to_string(index=False),
        timeline=[{"tool": "dataframe_operation_dtype_inspection", "status": "ok", "columns": len(rows)}],
        artifacts=[{"artifact_type": "table", "title": "Dtype inspection", "content": rows, "visibility": "user", "metadata": {"analysis_type": "dataframe_operation", "operation": "dtype_inspection"}}],
        trace_metadata={"fallback": "dataframe_operation", "analysis_type": "dataframe_operation", "operation": "dtype_inspection", "suppress_key_findings": True},
    )


def _parse_derived_formula(question: str, df: pd.DataFrame) -> dict[str, str] | None:
    match = re.search(r"(?:create|add|derive|создай|добавь)\s+`?([A-Za-z_][\w\s]*)`?\s*=\s*`?([^`=+*/-]+?)`?\s*([*])\s*`?([^`=+*/-]+?)`?(?:$|[,.]| and | и )", question, flags=re.IGNORECASE)
    if not match:
        return None
    target = match.group(1).strip().replace(" ", "_")
    left = _resolve_column_name(match.group(2), df)
    operator = match.group(3).strip()
    right = _resolve_column_name(match.group(4), df)
    if not target or not left or not right:
        return None
    return {"column": target, "left": left, "operator": operator, "right": right}


def _derived_column_response(question: str, df: pd.DataFrame, formula: dict[str, str]) -> dict[str, Any]:
    target = formula["column"]
    left = formula["left"]
    right = formula["right"]
    values = pd.to_numeric(df[left], errors="coerce") * pd.to_numeric(df[right], errors="coerce")
    preview = df[[left, right]].copy()
    preview[target] = values
    failures = int(values.isna().sum())
    rows = preview.head(10).to_dict(orient="records")
    summary = f"Created derived column `{target}` as `{left}` * `{right}` for dataframe-operation execution. {len(df) - failures:,} rows produced numeric values; {failures:,} rows could not be computed."
    lineage = {**formula, "source_columns": [left, right], "operation": "derived_column"}
    return _output(
        question=question,
        summary=summary,
        findings=[summary],
        evidence=[f"Computed `{target}` from `{left}` and `{right}` on the current dataframe."],
        limitations=["The derived column is returned with lineage metadata; persistence depends on the investigation state carrying `derived_columns` into the next run."],
        next_steps=[f"Use `{target}` as a metric in the next analytical query."],
        code=f"df[{target!r}] = pd.to_numeric(df[{left!r}], errors='coerce') * pd.to_numeric(df[{right!r}], errors='coerce')",
        result_preview=preview.head(10).to_string(index=False),
        timeline=[{"tool": "dataframe_operation_create_derived_column", "status": "ok", "derived_column": target}],
        artifacts=[{"artifact_type": "table", "title": f"Derived column {target}", "content": rows, "visibility": "user", "metadata": {"analysis_type": "dataframe_operation", "operation": "derived_column", "derived_column": target, "lineage": lineage}}],
        trace_metadata={"fallback": "dataframe_operation", "analysis_type": "dataframe_operation", "operation": "derived_column", "derived_column": target, "derived_columns": [lineage], "metric": target, "suppress_key_findings": True},
    )


def _parse_formula_validation(question: str, df: pd.DataFrame) -> dict[str, Any] | None:
    match = re.search(r"(?:check|validate|verify|проверь)\s+(?:whether\s+)?`?([^`=]+?)`?\s+(?:equals|=|равн)\s+([0-9]+(?:\.[0-9]+)?)\s*%\s+(?:of\s+)?`?([^`.,]+)`?", question, flags=re.IGNORECASE)
    if not match:
        return None
    target = _resolve_column_name(match.group(1), df)
    source = _resolve_column_name(match.group(3), df)
    if not target or not source:
        return None
    return {"target": target, "source": source, "rate": float(match.group(2)) / 100.0}


def _formula_validation_response(question: str, df: pd.DataFrame, validation: dict[str, Any]) -> dict[str, Any]:
    target = str(validation["target"])
    source = str(validation["source"])
    rate = float(validation["rate"])
    actual = pd.to_numeric(df[target], errors="coerce")
    expected = pd.to_numeric(df[source], errors="coerce") * rate
    diff = (actual - expected).abs()
    valid = diff <= 1e-9
    rows = pd.DataFrame({"actual": actual, "expected": expected, "absolute_difference": diff, "matches": valid}).head(20).to_dict(orient="records")
    mismatches = int((~valid.fillna(False)).sum())
    summary = f"Formula validation checked whether `{target}` equals {rate:.1%} of `{source}`. Mismatches: {mismatches:,} of {len(df):,} rows."
    return _output(
        question=question,
        summary=summary,
        findings=[summary],
        evidence=[f"Computed expected `{target}` as `{source}` * {rate:.4f} and compared row by row."],
        limitations=["This uses exact numeric tolerance; currency rounding rules may require a wider tolerance."],
        next_steps=["Review mismatch rows before using the formula-dependent field."],
        code=f"expected = df[{source!r}] * {rate}; diff = abs(df[{target!r}] - expected)",
        result_preview=pd.DataFrame(rows).to_string(index=False),
        timeline=[{"tool": "dataframe_operation_formula_validation", "status": "ok", "mismatches": mismatches}],
        artifacts=[{"artifact_type": "table", "title": f"Formula validation: {target}", "content": rows, "visibility": "user", "metadata": {"analysis_type": "dataframe_operation", "operation": "formula_validation", "target": target, "source": source, "mismatch_count": mismatches}}],
        trace_metadata={"fallback": "dataframe_operation", "analysis_type": "dataframe_operation", "operation": "formula_validation", "suppress_key_findings": True},
    )


def _category_inspection_response(question: str, df: pd.DataFrame) -> dict[str, Any]:
    mentioned = _mentioned_columns([str(col) for col in df.columns], question)
    columns = mentioned or [str(col) for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])][:10]
    rows = [{"column": column, "unique_values": int(df[column].nunique(dropna=True)), "missing": int(df[column].isna().sum())} for column in columns if column in df.columns]
    summary = f"Category inspection checked {len(rows)} column{'s' if len(rows) != 1 else ''}; no grouped metric ranking was created."
    return _output(
        question=question,
        summary=summary,
        findings=[summary],
        evidence=["Computed unique counts and missing values for category-like columns."],
        limitations=[],
        next_steps=[],
        code="df[column].nunique(dropna=True)",
        result_preview=pd.DataFrame(rows).to_string(index=False),
        timeline=[{"tool": "dataframe_operation_category_inspection", "status": "ok"}],
        artifacts=[{"artifact_type": "table", "title": "Category inspection", "content": rows, "visibility": "user", "metadata": {"analysis_type": "dataframe_operation", "operation": "category_inspection"}}],
        trace_metadata={"fallback": "dataframe_operation", "analysis_type": "dataframe_operation", "operation": "category_inspection", "suppress_key_findings": True},
    )


def _resolve_column_name(raw: str, df: pd.DataFrame) -> str:
    normalized = _normalize(str(raw).strip(" `"))
    for column in df.columns:
        if _normalize(str(column)) == normalized:
            return str(column)
    for column in df.columns:
        if normalized and normalized in _normalize(str(column)):
            return str(column)
    return ""


def _dataset_overview_summary(
    *,
    name: str,
    row_count: int,
    column_count: int,
    metrics: list[str],
    dimensions: list[str],
    timestamps: list[str],
    caveats: list[str],
) -> str:
    subject = _dataset_subject_hint(metrics, dimensions, timestamps)
    parts = [
        f"`{name}` looks like {subject}." if subject else f"`{name}` is ready for analytical exploration.",
        f"It contains {row_count:,} records and {column_count:,} columns." if row_count or column_count else "",
    ]
    if metrics or dimensions or timestamps:
        detail = []
        if metrics:
            detail.append(f"quantitative fields such as {', '.join(metrics[:4])}")
        if dimensions:
            detail.append(f"grouping fields such as {', '.join(dimensions[:5])}")
        if timestamps:
            detail.append(f"time fields such as {', '.join(timestamps[:3])}")
        parts.append("It includes " + "; ".join(detail) + ".")
    directions = _analytical_directions(metrics, dimensions, timestamps)
    if directions:
        parts.append("Strong starting directions are: " + "; ".join(directions[:5]) + ".")
    if metrics and dimensions:
        parts.append(f"A useful first analysis would be comparing `{metrics[0]}` across `{dimensions[0]}`.")
    elif metrics:
        parts.append(f"A useful first analysis would be checking the distribution and outliers of `{metrics[0]}`.")
    if caveats:
        parts.append(f"One data-quality caveat to check first: {caveats[0]}.")
    return " ".join(part for part in parts if part)


def _authoritative_plan_output(question: str, df: pd.DataFrame, conversation_context: dict[str, Any]) -> dict[str, Any] | None:
    plan = AuthoritativeExecutionPlanner.plan(
        question,
        df,
        active_branch=(conversation_context.get("conversation_state") or {}) if isinstance(conversation_context, dict) else {},
    )
    workspace = BranchWorkspaceManager.from_payload(
        ((conversation_context.get("conversation_state") or {}).get("branch_workspace") if isinstance(conversation_context, dict) else None)
    )
    route = BranchWorkspaceManager.route(plan, workspace)
    if plan.intent == "meta":
        return None
    if plan.hypothesis:
        return None
    if is_transformation_question(question) or _is_hypothesis_validation_question(question):
        return None
    if plan.requires_user_confirmation and plan.intent in {"rank_groups", "extremum", "filtered_minmax", "constrained_aggregation"}:
        explicit_dimension = _explicit_dimension_column(df, question, exclude={plan.metric} if plan.metric else set())
        if plan.metric and explicit_dimension and explicit_dimension != plan.metric:
            plan = replace(
                plan,
                dimension=explicit_dimension,
                dimension_source="explicit_column",
                requested_dimension_type=None,
                requires_user_confirmation=False,
                confirmation_reason=None,
                confidence="high",
            )
    if plan.requires_user_confirmation and plan.intent in {"rank_groups", "extremum", "filtered_minmax", "constrained_aggregation"}:
        requested = plan.requested_dimension_type or "requested grouping"
        alternatives = [item for item in plan.available_dimension_alternatives if item != plan.metric][:5]
        if alternatives:
            alt_text = ", ".join(f"`{item}`" for item in alternatives)
            summary = (
                f"I could not find a `{requested}`-like field in this dataset. "
                f"Available grouping fields include {alt_text}. Choose one of these, or upload a dataset with that field."
            )
        else:
            summary = (
                f"I could not find a `{requested}`-like grouping field in this dataset. "
                "Upload a dataset with that field before running this grouped analysis."
            )
        return _plan_output(
            question,
            summary,
            plan,
            route,
            findings=[],
            limitations=[f"The requested grouping target `{requested}` did not resolve to a schema field with sufficient confidence."],
        )
    direct = DirectQueryExecutor.execute(question=question, df=df, plan=plan, conversation_context=conversation_context)
    if direct:
        direct.setdefault("trace_metadata", {})["branch_route"] = route.to_payload() if hasattr(route, "to_payload") else {}
        direct["trace_metadata"]["branch_id"] = route.branch_id
        for artifact in direct.get("artifacts", []) if isinstance(direct.get("artifacts"), list) else []:
            if not isinstance(artifact, dict):
                continue
            metadata = artifact.get("metadata") if isinstance(artifact.get("metadata"), dict) else {}
            metadata["branch_id"] = route.branch_id
            metadata.setdefault("branch_type", _branch_type_from_route(route))
            artifact["metadata"] = metadata
        return direct
    if plan.intent == "bin_question":
        metric_name = plan.metric or _select_metric_column(df, question) or "the metric"
        return _plan_output(
            question,
            f"There is no saved `bin` field for this branch yet. Create bins from `{metric_name}` first, then compare record volume across those bins.",
            plan,
            route,
            findings=[f"`{metric_name}` needs explicit bins before bin-level comparisons."],
        )
    if plan.intent == "binning":
        return _execute_binning_plan(question, df, plan, route)
    if plan.intent == "filtered_minmax":
        return _execute_filtered_minmax_plan(question, df, plan, route)
    if plan.intent == "shipping_delay":
        return _execute_shipping_delay_plan(question, df, plan, route)
    if plan.intent == "rank_groups" and plan.metric and plan.dimension:
        return _execute_rank_groups_plan(question, df, plan, route)
    if plan.intent == "histogram":
        return _execute_histogram_plan(question, df, plan, route)
    if plan.intent == "seasonality_heatmap":
        return _execute_seasonality_heatmap_plan(question, df, plan, route)
    return None


def _branch_type_from_route(route: Any) -> str:
    identity = getattr(route, "identity", None)
    intent = str(getattr(identity, "intent", "") or "")
    chart_type = str(getattr(identity, "chart_type", "") or "")
    if intent == "shipping_delay":
        return "shipping"
    if intent in {"histogram", "binning"} or chart_type == "histogram":
        return "distribution"
    if intent in {"temporal_trend", "chart_request", "growth", "seasonality_heatmap"} or chart_type in {"line", "heatmap", "seasonality_heatmap"}:
        return "temporal"
    if getattr(identity, "dimension", None):
        return "grouped"
    return intent or "generic"


def _plan_output(
    question: str,
    summary: str,
    plan: AuthoritativeQueryPlan,
    route: Any,
    *,
    findings: list[str] | None = None,
    artifacts: list[dict[str, Any]] | None = None,
    result_preview: str = "",
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    route_payload = route.to_payload() if hasattr(route, "to_payload") else {}
    branch_id = str(getattr(route, "branch_id", "") or route_payload.get("branch_id") or "")
    prepared_artifacts = artifacts or []
    for artifact in prepared_artifacts:
        if not isinstance(artifact, dict):
            continue
        metadata = artifact.get("metadata") if isinstance(artifact.get("metadata"), dict) else {}
        if branch_id:
            metadata["branch_id"] = branch_id
        metadata.setdefault("branch_type", _branch_type_from_route(route))
        metadata.setdefault("metric", plan.metric)
        metadata.setdefault("dimension", plan.dimension)
        metadata.setdefault("aggregation", plan.aggregation)
        metadata.setdefault("filters", [item.to_payload() for item in plan.filters])
        content = artifact.get("content") if isinstance(artifact.get("content"), dict) else {}
        metadata.setdefault("chart_type", content.get("chart_type") or plan.chart_type)
        row_count = content.get("row_count")
        if row_count is None and isinstance(artifact.get("content"), list):
            row_count = len(artifact["content"])
        metadata.setdefault("row_count", row_count)
        metadata.setdefault("artifact_type", artifact.get("artifact_type") or artifact.get("type"))
        metadata.setdefault("query_plan", plan.to_payload())
        artifact["metadata"] = metadata
    return _output(
        question=question,
        summary=summary,
        findings=findings or [summary],
        evidence=[
            f"Authoritative plan: intent={plan.intent}, metric={plan.metric}, dimension={plan.dimension}, chart={plan.chart_type}.",
            f"Branch route: {getattr(route, 'action', '')}.",
        ],
        limitations=limitations or [],
        next_steps=[],
        code="",
        result_preview=result_preview,
        timeline=[
            {"tool": "deterministic_pandas_fallback", "status": "ok", "intent": plan.intent},
            {"tool": "authoritative_execution_planner", "status": "ok", "intent": plan.intent},
            {"tool": "branch_workspace_manager", "status": "ok", "action": getattr(getattr(route, "action", None), "value", str(getattr(route, "action", "")))},
        ],
        artifacts=prepared_artifacts,
        trace_metadata={
            "fallback": "authoritative_query_plan",
            "analysis_type": plan.intent,
            "metric": plan.metric,
            "dimension": plan.dimension,
            "chart_type": plan.chart_type,
            "query_plan": plan.to_payload(),
            "branch_id": branch_id,
            "branch_route": route_payload,
        },
    )


def _execute_rank_groups_plan(question: str, df: pd.DataFrame, plan: AuthoritativeQueryPlan, route: Any) -> dict[str, Any] | None:
    if not plan.metric or not plan.dimension or plan.metric not in df.columns or plan.dimension not in df.columns:
        return None
    working = _apply_plan_filters(df, plan)
    if working.empty:
        return _plan_output(question, "The requested filters leave no rows, so the ranking cannot be computed.", plan, route)
    metric = pd.to_numeric(working[plan.metric], errors="coerce")
    grouped = (
        working.assign(**{plan.metric: metric})
        .dropna(subset=[plan.metric])
        .groupby(plan.dimension, dropna=False)[plan.metric]
        .agg(total="sum", mean="mean", count="count")
        .reset_index()
        .sort_values("total" if plan.aggregation == "sum" else "mean", ascending=plan.ranking_direction == "ascending")
    )
    if grouped.empty:
        return None
    rows = grouped.head(20).to_dict(orient="records")
    top = rows[:5]
    value_key = "total" if plan.aggregation == "sum" else "mean"
    aggregation_label = "total" if plan.aggregation == "sum" else "average"
    leader_bits = ", ".join(f"`{row[plan.dimension]}` ({aggregation_label} {float(row[value_key]):.2f}, n={int(row['count'])})" for row in top)
    filter_text = _filter_text(plan)
    alias_text = f"`{plan.metric_alias_used}` is treated as `{plan.metric}`. " if plan.metric_alias_used and str(plan.metric_alias_used).casefold() != str(plan.metric).casefold() else ""
    summary = f"{alias_text}{filter_text}`{plan.metric}` by `{plan.dimension}` is led by {leader_bits} using {aggregation_label} `{plan.metric}`."
    chart_rows = grouped.head(20)[[plan.dimension, "total", "mean", "count"]].to_dict(orient="records")
    artifact_title = f"Average {plan.metric} by {plan.dimension}" if plan.aggregation == "mean" else f"{plan.metric} by {plan.dimension}"
    return _plan_output(
        question,
        summary,
        plan,
        route,
        findings=[summary],
        artifacts=[
            {"artifact_type": "table", "title": artifact_title, "content": rows, "visibility": "user", "metadata": {"query_plan": plan.to_payload()}},
            {
                "artifact_type": "chart",
                "title": artifact_title,
                "content": {
                    "chart_type": "bar",
                    "x": plan.dimension,
                    "y": value_key,
                    "metric": plan.metric,
                    "dimension": plan.dimension,
                    "aggregation": plan.aggregation or "sum",
                    "filters": [item.to_payload() for item in plan.filters],
                    "rows": chart_rows,
                    "ranking_scope": f"Top {min(len(chart_rows), 20)} `{plan.dimension}` groups by {aggregation_label} `{plan.metric}`.",
                    "row_count": int(len(working)),
                },
                "visibility": "user",
                "pinned": True,
                "metadata": {
                    "query_plan": plan.to_payload(),
                    "metric": plan.metric,
                    "dimension": plan.dimension,
                    "aggregation": plan.aggregation or "sum",
                    "filters": [item.to_payload() for item in plan.filters],
                    "ranking_mode": plan.ranking_direction or "descending",
                    "transformation_state": plan.transformation or "raw",
                    "chart_type": "bar",
                    "sorted_values": chart_rows,
                    "top_n": int(min(len(chart_rows), 20)),
                    "row_count": int(len(working)),
                },
            },
        ],
        result_preview=pd.DataFrame(rows).to_string(index=False),
    )


def _execute_filtered_minmax_plan(question: str, df: pd.DataFrame, plan: AuthoritativeQueryPlan, route: Any) -> dict[str, Any] | None:
    if not plan.metric or plan.metric not in df.columns:
        return None
    customer_col = _semantic_column_by_markers(df, ("customer", "client", "клиент", "buyer"))
    if not customer_col:
        return None
    errors = FilterConstraintValidator.validate(plan, df)
    if errors:
        return _plan_output(question, "The requested filter could not be applied: " + "; ".join(errors), plan, route, limitations=errors)
    working = _apply_plan_filters(df, plan)
    metric = pd.to_numeric(working[plan.metric], errors="coerce")
    grouped = (
        working.assign(**{plan.metric: metric})
        .dropna(subset=[customer_col, plan.metric])
        .groupby(customer_col, dropna=False)[plan.metric]
        .sum()
        .reset_index(name=f"total_{plan.metric}")
        .sort_values(f"total_{plan.metric}", ascending=True)
    )
    if grouped.empty:
        return _plan_output(question, "No matching customer rows remain after applying the requested filters.", plan, route)
    row = grouped.iloc[0]
    filter_columns = ", ".join(f"`{item.column}`" for item in plan.filters) or "requested"
    summary = f"Within {_filter_text(plan).strip()}, `{row[customer_col]}` has the lowest total `{plan.metric}` at {float(row[f'total_{plan.metric}']):.2f}. The {filter_columns} filter is preserved; this is not a global `{customer_col}` ranking."
    rows = grouped.head(20).to_dict(orient="records")
    return _plan_output(question, summary, plan, route, findings=[summary], artifacts=[{"artifact_type": "table", "title": f"Filtered customer {plan.metric}", "content": rows, "visibility": "user"}], result_preview=pd.DataFrame(rows).to_string(index=False))


def _execute_histogram_plan(question: str, df: pd.DataFrame, plan: AuthoritativeQueryPlan, route: Any) -> dict[str, Any] | None:
    if not plan.metric or plan.metric not in df.columns:
        return None
    working = _apply_plan_filters(df, plan)
    values = pd.to_numeric(working[plan.metric], errors="coerce").dropna()
    if values.empty:
        return _plan_output(question, "The histogram cannot be built because no numeric values remain after filtering.", plan, route)
    counts = pd.cut(values, bins=min(10, max(3, int(values.nunique())))).value_counts().sort_index()
    rows = [
        {"bin": f"{index.left:.2f} to {index.right:.2f}", "label": f"{index.left:.2f} to {index.right:.2f}", "count": int(count), "left": float(index.left), "right": float(index.right)}
        for index, count in counts.items()
    ]
    filter_text = _filter_text(plan)
    summary = f"`{plan.metric}` distribution {filter_text}uses {len(values):,} rows. Median is {float(values.median()):.2f}, mean is {float(values.mean()):.2f}, and the range is {float(values.min()):.2f} to {float(values.max()):.2f}."
    title = f"{plan.metric} distribution" + (f" in {plan.filters[0].value}" if len(plan.filters) == 1 else "")
    artifact = build_histogram_artifact(metric=plan.metric, bins=rows, filters=plan.filters, title=title, query_plan=plan.to_payload(), row_count=int(len(values)))
    return _plan_output(question, summary, plan, route, findings=[summary], artifacts=[artifact], result_preview=pd.DataFrame(rows).to_string(index=False))


def _execute_seasonality_heatmap_plan(question: str, df: pd.DataFrame, plan: AuthoritativeQueryPlan, route: Any) -> dict[str, Any] | None:
    if not plan.metric or not plan.time_axis or plan.metric not in df.columns or plan.time_axis not in df.columns:
        return _plan_output(question, "A seasonality heatmap needs a numeric metric and a date column; I will not replace the requested heatmap with a grouped bar chart.", plan, route)
    working = df[[plan.time_axis, plan.metric]].copy()
    working[plan.time_axis] = pd.to_datetime(working[plan.time_axis], errors="coerce")
    working[plan.metric] = pd.to_numeric(working[plan.metric], errors="coerce")
    working = working.dropna()
    if working.empty:
        return _plan_output(question, "A seasonality heatmap cannot be built because the date or metric values could not be parsed.", plan, route)
    working["year"] = working[plan.time_axis].dt.year
    working["month"] = working[plan.time_axis].dt.month
    pivot = working.groupby(["year", "month"])[plan.metric].mean().reset_index(name="mean")
    rows = pivot.to_dict(orient="records")
    summary = f"Seasonality heatmap plan preserved: using monthly average `{plan.metric}` by year and month from `{plan.time_axis}`. The strongest cell is year {int(pivot.loc[pivot['mean'].idxmax(), 'year'])}, month {int(pivot.loc[pivot['mean'].idxmax(), 'month'])} with average {float(pivot['mean'].max()):.2f}."
    return _plan_output(question, summary, plan, route, findings=[summary], artifacts=[{"artifact_type": "chart", "title": f"{plan.metric} seasonality heatmap", "content": {"chart_type": "heatmap", "visualization_type": "seasonality_heatmap", "x": "month", "y": "year", "value": "mean", "rows": rows}, "visibility": "user", "pinned": True, "metadata": {"branch_type": "temporal", "query_plan": plan.to_payload()}}], result_preview=pd.DataFrame(rows).to_string(index=False))


def _execute_binning_plan(question: str, df: pd.DataFrame, plan: AuthoritativeQueryPlan, route: Any) -> dict[str, Any] | None:
    metric = plan.metric if plan.metric in df.columns else _select_metric_column(df, question)
    if not metric:
        return None
    values = pd.to_numeric(df[metric], errors="coerce").dropna()
    bins = pd.qcut(values, q=min(4, max(2, values.nunique())), duplicates="drop")
    rows = bins.value_counts().sort_index().reset_index()
    rows.columns = [f"{metric}_bin", "record_count"]
    records = rows.to_dict(orient="records")
    summary = f"Created `{metric}_bin` from `{metric}` using quantile bins. The bins are ready for contribution, sparsity, and volume-vs-value checks."
    derived = {
        "derived_field": f"{metric}_bin",
        "source_metric": metric,
        "method": "quantile",
        "labels": [str(item) for item in rows[f"{metric}_bin"].tolist()],
    }
    result = _plan_output(question, summary, plan, route, findings=[summary], artifacts=[{"artifact_type": "table", "title": f"{metric} automatic bins", "content": records, "visibility": "user", "metadata": {"derived_field": derived, "branch_type": "distribution", "query_plan": plan.to_payload()}}], result_preview=rows.to_string(index=False))
    result["trace_metadata"]["derived_field"] = derived
    return result


def _distribution_comparison_followup(question: str, df: pd.DataFrame, conversation_context: dict[str, Any]) -> dict[str, Any] | None:
    normalized = _normalize(question)
    if not any(marker in normalized for marker in ("compare against", "compare with", "сравн")):
        return None
    chart = conversation_context.get("latest_chart_context") if isinstance(conversation_context.get("latest_chart_context"), dict) else {}
    state = conversation_context.get("conversation_state") if isinstance(conversation_context.get("conversation_state"), dict) else {}
    distribution_state = state.get("distribution_state") if isinstance(state.get("distribution_state"), dict) else {}
    if chart.get("chart_type") != "histogram" and distribution_state.get("chart_type") != "histogram":
        return None
    metric = str(chart.get("metric") or "")
    if not metric:
        metric = str(distribution_state.get("metric") or "")
    if not metric or metric not in df.columns:
        return None
    filters = []
    content_filters = distribution_state.get("filters") if isinstance(distribution_state.get("filters"), list) else []
    recent = conversation_context.get("recent_artifacts") if isinstance(conversation_context.get("recent_artifacts"), list) else []
    if not content_filters:
        for artifact in reversed(recent):
            content = artifact.get("content") if isinstance(artifact, dict) else None
            if isinstance(content, dict) and content.get("chart_type") == "histogram":
                content_filters = content.get("filters") if isinstance(content.get("filters"), list) else []
                break
    base_filter = next((item for item in content_filters if isinstance(item, dict) and item.get("column") in df.columns), None)
    if not base_filter:
        return None
    filter_col = str(base_filter.get("column"))
    base_value = str(base_filter.get("value"))
    target_value = _matched_value_from_question(question, df, filter_col, exclude={base_value})
    if not target_value:
        return None
    rows = []
    group_values: dict[str, pd.Series] = {}
    for value in (base_value, target_value):
        subset = df[df[filter_col].astype(str).str.casefold() == value.casefold()]
        values = pd.to_numeric(subset[metric], errors="coerce").dropna()
        group_values[value] = values
        if values.empty:
            rows.append({filter_col: value, "n": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0})
        else:
            rows.append({filter_col: value, "n": int(len(values)), "mean": float(values.mean()), "median": float(values.median()), "min": float(values.min()), "max": float(values.max())})
    if any(int(row.get("n") or 0) <= 0 for row in rows):
        return _output(
            question=question,
            summary=f"The distribution comparison cannot be charted because one requested `{filter_col}` group has no numeric `{metric}` rows.",
            findings=[],
            evidence=[],
            limitations=["Comparison histograms require positive row counts for every comparison group."],
            next_steps=[],
            code="",
            result_preview=pd.DataFrame(rows).to_string(index=False),
            timeline=[{"tool": "distribution_comparison", "status": "blocked"}],
            artifacts=[],
            trace_metadata={"fallback": "distribution_comparison", "analysis_type": "artifact_validation_failed", "metric": metric, "dimension": filter_col, "chart_type": "histogram"},
        )
    summary = (
        f"`{metric}` in `{base_value}` is compared against `{target_value}` by `{filter_col}`. "
        f"`{base_value}`: n={rows[0]['n']}, mean {rows[0]['mean']:.2f}, median {rows[0]['median']:.2f}, range {rows[0]['min']:.2f} to {rows[0]['max']:.2f}. "
        f"`{target_value}`: n={rows[1]['n']}, mean {rows[1]['mean']:.2f}, median {rows[1]['median']:.2f}, range {rows[1]['min']:.2f} to {rows[1]['max']:.2f}. "
        "Read this as a distribution comparison: the median, mean, and range separate center shift from tail effects."
    )
    row_count = int(sum(int(row.get("n") or 0) for row in rows))
    combined_values = pd.concat([series for series in group_values.values() if not series.empty])
    _, shared_edges = pd.cut(combined_values, bins=min(10, max(3, int(combined_values.nunique()))), retbins=True, duplicates="drop")
    comparison_groups = [
        {
            "group": str(row.get(filter_col)),
            "label": str(row.get(filter_col)),
            "record_count": int(row.get("n") or 0),
            "n": int(row.get("n") or 0),
            "bins": [
                {
                    "left": float(interval.left),
                    "right": float(interval.right),
                    "label": f"{interval.left:.2f} to {interval.right:.2f}",
                    "count": int(count),
                }
                for interval, count in pd.cut(group_values[str(row.get(filter_col))], bins=shared_edges, include_lowest=True).value_counts().sort_index().items()
            ],
            "mean": float(row.get("mean") or 0),
            "median": float(row.get("median") or 0),
            "min": float(row.get("min") or 0),
            "max": float(row.get("max") or 0),
        }
        for row in rows
    ]
    artifact_content = {
        "chart_type": "histogram",
        "visualization_type": "histogram",
        "metric": metric,
        "x": metric,
        "y": "record_count",
        "rows": rows,
        "series": filter_col,
        "filters": [base_filter],
        "comparison_groups": comparison_groups,
        "row_count": row_count,
        "distribution_summary": {
            "row_count": row_count,
            "group_count": len(comparison_groups),
            "comparison_dimension": filter_col,
            "groups": [base_value, target_value],
        },
    }
    return _output(
        question=question,
        summary=summary,
        findings=[summary],
        evidence=[f"Compared `{metric}` distributions for `{filter_col}` values `{base_value}` and `{target_value}`."],
        limitations=[],
        next_steps=[],
        code="",
        result_preview=pd.DataFrame(rows).to_string(index=False),
        timeline=[{"tool": "distribution_comparison", "status": "ok"}],
        artifacts=[{"artifact_type": "chart", "title": f"{metric} distribution comparison", "content": artifact_content, "visibility": "user", "metadata": {"branch_type": "distribution_comparison", "chart_type": "histogram", "metric": metric, "dimension": filter_col, "row_count": row_count}}],
        trace_metadata={"fallback": "distribution_comparison", "analysis_type": "distribution_comparison", "metric": metric, "dimension": filter_col, "chart_type": "histogram"},
    )


def _bins_followup_output(question: str, df: pd.DataFrame, conversation_context: dict[str, Any]) -> dict[str, Any] | None:
    normalized = _normalize(question)
    if "bin" not in normalized and "bins" not in normalized:
        return None
    state = conversation_context.get("conversation_state") if isinstance(conversation_context.get("conversation_state"), dict) else {}
    derived = state.get("derived_field") if isinstance(state.get("derived_field"), dict) else {}
    if not derived:
        return None
    metric = str(derived.get("source_metric") or _select_metric_column(df, question) or "")
    derived_field = str(derived.get("derived_field") or (f"{metric}_bin" if metric else ""))
    if not metric or metric not in df.columns or not derived_field:
        return None
    values = pd.to_numeric(df[metric], errors="coerce")
    bins = pd.qcut(values.dropna(), q=min(4, max(2, values.nunique())), duplicates="drop")
    binned = df.loc[values.dropna().index].copy()
    binned[derived_field] = bins.astype(str)
    grouped = (
        binned.groupby(derived_field, dropna=False)[metric]
        .agg(record_count="count", total="sum", average="mean")
        .reset_index()
    )
    rows = grouped.to_dict(orient="records")
    summary, analysis_type = _bin_followup_summary(question, rows, derived_field, metric)
    return _output(
        question=question,
        summary=summary,
        findings=[summary],
        evidence=[f"Used persisted derived field metadata for `{derived_field}` from `{metric}`."],
        limitations=[],
        next_steps=[],
        code="",
        result_preview=grouped.to_string(index=False),
        timeline=[{"tool": "derived_bins_followup", "status": "ok"}],
        artifacts=[{"artifact_type": "table", "title": f"{derived_field} analysis", "content": rows, "visibility": "user", "metadata": {"derived_field": derived, "branch_type": "distribution", "analysis_type": analysis_type}}],
        trace_metadata={"fallback": "derived_bins_followup", "analysis_type": analysis_type, "metric": metric, "dimension": derived_field, "derived_field": derived},
    )


def _bin_followup_summary(question: str, rows: list[dict[str, Any]], derived_field: str, metric: str) -> tuple[str, str]:
    normalized = _normalize(question)
    sorted_total = sorted(rows, key=lambda row: float(row.get("total") or 0), reverse=True)
    sorted_count = sorted(rows, key=lambda row: int(row.get("record_count") or 0), reverse=True)
    sorted_avg = sorted(rows, key=lambda row: float(row.get("average") or 0), reverse=True)
    top_total = sorted_total[0] if sorted_total else {}
    top_count = sorted_count[0] if sorted_count else {}
    top_avg = sorted_avg[0] if sorted_avg else {}
    if any(marker in normalized for marker in ("contribute", "contributes", "most revenue", "most sales", "largest total", "biggest total", "вклад")):
        return (
            f"`{top_total.get(derived_field)}` contributes the most `{metric}` with total {float(top_total.get('total') or 0):.2f} across {int(top_total.get('record_count') or 0)} rows.",
            "bins_contribution",
        )
    if any(marker in normalized for marker in ("driven", "explain", "volume", "order value", "average", "связ", "объяс", "объем", "объём")):
        return (
            f"Across `{derived_field}`, the largest total is `{top_total.get(derived_field)}` ({float(top_total.get('total') or 0):.2f}), the largest record volume is `{top_count.get(derived_field)}` (n={int(top_count.get('record_count') or 0)}), and the highest average `{metric}` is `{top_avg.get(derived_field)}` ({float(top_avg.get('average') or 0):.2f}). If the total leader matches the volume leader, volume is the main explanation; if it matches the average leader, order value is the stronger driver.",
            "bins_volume_value_driver",
        )
    if any(marker in normalized for marker in ("sparse", "small sample", "low sample", "few records", "малень", "редк")):
        counts = [int(row.get("record_count") or 0) for row in rows]
        if counts and max(counts) - min(counts) <= 1:
            return (f"No `{derived_field}` bins are materially sparse; the quantile strategy produced nearly balanced groups with {min(counts)} to {max(counts)} records per bin.", "bins_sparsity_balanced")
        sparse = sorted(rows, key=lambda row: int(row.get("record_count") or 0))[:3]
        bits = ", ".join(f"`{row.get(derived_field)}` n={int(row.get('record_count') or 0)}" for row in sparse)
        return (f"Sparsest `{derived_field}` bins are {bits}. These bins need caution before comparing `{metric}` averages.", "bins_sparsity")
    spread = float(top_avg.get("average") or 0) - float(sorted_avg[-1].get("average") or 0) if len(sorted_avg) > 1 else 0.0
    return (
        f"`{derived_field}` bins differ in both volume and average `{metric}`. Record-volume leader: `{top_count.get(derived_field)}` (n={int(top_count.get('record_count') or 0)}); average-value leader: `{top_avg.get(derived_field)}` ({float(top_avg.get('average') or 0):.2f}); average spread is {spread:.2f}.",
        "bins_comparison",
    )


def _matched_value_from_question(question: str, df: pd.DataFrame, column: str, exclude: set[str] | None = None) -> str:
    resolved = resolve_categorical_value(question, df, preferred_columns=[column], exclude=exclude)
    return resolved.value if resolved else ""


def _execute_shipping_delay_plan(question: str, df: pd.DataFrame, plan: AuthoritativeQueryPlan, route: Any) -> dict[str, Any] | None:
    metric = plan.metric if plan.metric in df.columns else _select_metric_column(df, question)
    order_col = _semantic_column_by_markers(df, ORDER_DATE_MARKERS)
    ship_col = _semantic_column_by_markers(df, DELIVERY_DATE_MARKERS)
    if not metric or not order_col or not ship_col:
        return _plan_output(question, "A shipping-delay relationship check needs a numeric metric, an order-like date column, and a shipping-like date column; I will not replace it with a generic trend summary.", plan, route)
    behavior_col = _semantic_column_by_markers(df, SHIPPING_METHOD_MARKERS)
    delay_analysis = compute_delivery_delay_analysis(df, metric=metric, order_column=order_col, delivery_column=ship_col, method_column=behavior_col)
    selected_columns = [metric, order_col, ship_col] + ([behavior_col] if behavior_col else [])
    working = df[selected_columns].copy()
    working[metric] = pd.to_numeric(working[metric], errors="coerce")
    working[order_col] = pd.to_datetime(working[order_col], errors="coerce")
    working[ship_col] = pd.to_datetime(working[ship_col], errors="coerce")
    working = working.dropna(subset=[metric, order_col, ship_col])
    if working.empty:
        return _plan_output(question, "Shipping-delay relationship could not be checked because dates or metric values did not parse.", plan, route)
    working["delivery_delay_days"] = (working[ship_col] - working[order_col]).dt.days
    issues = TemporalSanityValidator.delivery_delay_issues(working, order_col, ship_col)
    working["period"] = working[order_col].dt.to_period("M").dt.to_timestamp()
    monthly = working.groupby("period")[metric].mean().reset_index(name="mean")
    monthly["change"] = monthly["mean"].diff()
    positive = monthly[monthly["change"] > 0]
    spike_periods = set(positive.nlargest(min(3, len(positive)), "change")["period"]) if not positive.empty else set()
    working["is_growth_period"] = working["period"].isin(spike_periods)
    spike_delay = float(working.loc[working["is_growth_period"], "delivery_delay_days"].mean()) if working["is_growth_period"].any() else 0.0
    normal_delay = float(working.loc[~working["is_growth_period"], "delivery_delay_days"].mean()) if (~working["is_growth_period"]).any() else 0.0
    gap = spike_delay - normal_delay
    rows = [
        {"period_type": "growth/spike", "avg_delay_days": spike_delay, "rows": int(working["is_growth_period"].sum())},
        {"period_type": "other", "avg_delay_days": normal_delay, "rows": int((~working["is_growth_period"]).sum())},
    ]
    sanity = "Parsed delays look consistent." if not issues else "Some parsed delays look unusual: " + "; ".join(issues) + ". Interpret this cautiously."
    summary = f"Using monthly average `{metric}` growth periods, I derived `delivery_delay_days` from `{ship_col}` minus `{order_col}`. {sanity} Growth-period average delay is {spike_delay:.2f} days versus {normal_delay:.2f} outside growth periods, a gap of {gap:+.2f} days."
    artifacts = [{"artifact_type": "table", "title": "Shipping delay relationship", "content": rows, "visibility": "user", "metadata": {"branch_type": "shipping", "query_plan": plan.to_payload()}}]
    if delay_analysis:
        artifacts.extend(build_delivery_delay_artifacts(delay_analysis, query_plan=plan.to_payload()))
    return _plan_output(
        question,
        summary,
        plan,
        route,
        findings=[summary],
        artifacts=artifacts,
        result_preview=pd.DataFrame(rows).to_string(index=False),
        limitations=["This is temporal association, not causal proof. Control for the shipping-method field before treating delay as a driver."],
    )


def _apply_plan_filters(df: pd.DataFrame, plan: AuthoritativeQueryPlan) -> pd.DataFrame:
    working = df.copy()
    for item in plan.filters:
        if item.column in working.columns and item.operator == "equals":
            working = working[working[item.column].astype(str).str.casefold() == str(item.value).casefold()]
    return working


def _filter_text(plan: AuthoritativeQueryPlan) -> str:
    if not plan.filters:
        return ""
    return "with " + ", ".join(f"`{item.column}` = `{item.value}`" for item in plan.filters) + " "


def _is_branch_switch_to_dimension(question: str, dimension_col: str) -> bool:
    text = _normalize(question)
    dimension = _normalize(dimension_col)
    switch_markers = ("switch back", "back to", "переключ", "верни")
    return any(marker in text for marker in switch_markers) and (
        dimension in text
        or ("city" in dimension and any(marker in text for marker in ("city", "cities", "город", "города")))
    )


def _dataframe_overview_response(question: str, df: pd.DataFrame) -> dict[str, Any]:
    semantic = build_semantic_dataset_profile(df=df)
    metrics = [candidate["name"] for candidate in _rank_metric_candidates(df, semantic)[:6]]
    dimensions = [column.name for column in semantic.dimensions]
    timestamps = [column.name for column in semantic.timestamps]
    missing = df.isna().sum()
    missing_columns = [str(column) for column, count in missing.items() if int(count) > 0]
    caveats = []
    if missing_columns:
        caveats.append(f"missing values appear in {', '.join(missing_columns[:4])}")
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        caveats.append(f"{duplicate_count} duplicate rows were detected")
    summary = _dataset_overview_summary(
        name="the dataset",
        row_count=len(df),
        column_count=len(df.columns),
        metrics=metrics,
        dimensions=dimensions,
        timestamps=timestamps,
        caveats=caveats,
    )
    timeline = _timeline("dataset_overview", rows=len(df), columns=len(df.columns))
    return _output(
        question=question,
        summary=summary,
        findings=[],
        evidence=[f"Dataset overview based on {len(df):,} rows and {len(df.columns):,} columns."],
        limitations=[
            "This overview is descriptive and should be validated with focused comparisons, trends, correlations, outlier checks, or charts.",
        ],
        next_steps=_analytical_directions(metrics, dimensions, timestamps)[:5],
        code="",
        result_preview="",
        timeline=timeline,
        artifacts=[],
        trace_metadata={"fallback": "dataset_overview", "analysis_type": "overview", "suppress_key_findings": True},
    )


def _dataframe_exploration_response(question: str, df: pd.DataFrame) -> dict[str, Any]:
    semantic = build_semantic_dataset_profile(df=df)
    ranked_metrics = _rank_metric_candidates(df, semantic)
    metrics = [item["name"] for item in ranked_metrics[:6]]
    dimensions = [column.name for column in semantic.dimensions[:8]]
    timestamps = [column.name for column in semantic.timestamps[:4]]
    excluded = [column.name for column in semantic.identifiers[:6]]
    missing = df.isna().sum()
    missing_columns = [str(column) for column, count in missing.items() if int(count) > 0]
    parts = [
        f"This is a structured tabular dataset with {len(df):,} rows and {len(df.columns):,} columns.",
    ]
    if metrics:
        metric_bits = []
        for item in ranked_metrics[:3]:
            metric_bits.append(f"`{item['name']}` ({item['reason']})")
        parts.append("The most useful numeric measures for analysis are: " + "; ".join(metric_bits) + ".")
    if dimensions:
        parts.append("Comparison fields: " + ", ".join(f"`{item}`" for item in dimensions[:6]) + ".")
    if timestamps:
        parts.append("Time-based analysis can use: " + ", ".join(f"`{item}`" for item in timestamps[:3]) + ".")
    if excluded:
        parts.append("I will not use identifier or code-like fields as default metrics: " + ", ".join(f"`{item}`" for item in excluded[:5]) + ".")
    if missing_columns:
        parts.append("Data-quality note: missing values appear in " + ", ".join(f"`{item}`" for item in missing_columns[:5]) + ".")
    if metrics and dimensions:
        parts.append(f"A useful first analysis would be comparing `{metrics[0]}` across `{dimensions[0]}`.")
    elif metrics:
        parts.append(f"A useful first analysis would be checking the distribution and outliers of `{metrics[0]}`.")
    next_steps = _analytical_directions(metrics, dimensions, timestamps)[:5]
    if next_steps:
        parts.append("Strong starting directions are: " + "; ".join(next_steps[:5]) + ".")
    return _output(
        question=question,
        summary=" ".join(parts),
        findings=[],
        evidence=[f"Scanned {len(df):,} rows and {len(df.columns):,} columns; ranked candidate metrics by semantic role, sparsity, cardinality, and distribution usefulness."],
        limitations=["This is dataset orientation, not a validated analytical finding."],
        next_steps=next_steps,
        code="",
        result_preview="",
        timeline=_timeline("dataset_overview", rows=len(df), columns=len(df.columns), scan="exploration"),
        artifacts=[],
        trace_metadata={"fallback": "dataset_exploration", "analysis_type": "overview", "suppress_key_findings": True},
    )


def _dataframe_overview_with_dependencies_response(question: str, df: pd.DataFrame) -> dict[str, Any]:
    response = _dataframe_exploration_response(question, df)
    semantic = build_semantic_dataset_profile(df=df)
    ranked_metrics = _rank_metric_candidates(df, semantic)
    metrics = [candidate["name"] for candidate in ranked_metrics]
    dimensions = [column.name for column in semantic.dimensions]
    timestamps = [column.name for column in semantic.timestamps]
    business_metrics = [metric for metric in metrics if metric in df.columns and not _looks_identifier_like(df[metric], metric)]
    primary_metric = _preferred_business_metric(business_metrics) or (business_metrics[0] if business_metrics else (metrics[0] if metrics else None))
    dependency_directions = _dependency_directions(metrics, dimensions, timestamps, primary_metric=primary_metric)
    relationship_evidence = _candidate_relationship_evidence(df, metrics, dimensions, timestamps)
    overview_summary = _concrete_dataset_scan_summary(df, metrics, dimensions, timestamps, primary_metric)
    if relationship_evidence:
        summary = (
            f"{overview_summary} For dependency checks, the main working metric here is `{primary_metric}`. "
            + " ".join(relationship_evidence[:3])
            + " The strongest next checks are: "
            + "; ".join(dependency_directions[:4])
            + "."
        )
        response["final_answer"] = summary
        response["summary"] = summary
        if isinstance(response.get("structured_report"), dict):
            response["structured_report"]["summary"] = summary
            response["structured_report"]["key_findings"] = relationship_evidence[:4]
            response["structured_report"]["next_steps"] = dependency_directions[:5]
        response["key_findings"] = relationship_evidence[:4]
        response["next_steps"] = dependency_directions[:5]
    elif dependency_directions:
        summary = (
            f"{overview_summary} The first dependencies should be tested as hypotheses rather than treated as finished findings: "
            + "; ".join(dependency_directions[:5])
            + "."
        )
        response["final_answer"] = summary
        response["summary"] = summary
        if isinstance(response.get("structured_report"), dict):
            response["structured_report"]["summary"] = summary
            response["structured_report"]["next_steps"] = dependency_directions[:5]
        response["next_steps"] = dependency_directions[:5]
    else:
        summary = (
            f"{overview_summary} Для поиска зависимостей нужен хотя бы один числовой показатель "
            "и одно поле сегментации или времени."
        )
        response["final_answer"] = summary
        if isinstance(response.get("structured_report"), dict):
            response["structured_report"]["summary"] = summary
    response["trace_metadata"] = {
        **(response.get("trace_metadata") or {}),
        "analysis_type": "overview_dependencies",
        "suppress_key_findings": not bool(relationship_evidence),
    }
    return response


def _concrete_dataset_scan_summary(
    df: pd.DataFrame,
    metrics: list[str],
    dimensions: list[str],
    timestamps: list[str],
    primary_metric: str | None,
) -> str:
    subject = _dataset_subject_hint(metrics, dimensions, timestamps)
    parts = [
        f"Данные похожи на {subject}: {len(df):,} строк и {len(df.columns):,} колонок.",
    ]
    if primary_metric and primary_metric in df.columns:
        series = pd.to_numeric(df[primary_metric], errors="coerce").dropna()
        if not series.empty:
            parts.append(
                f"По `{primary_metric}` диапазон широкий: median {float(series.median()):.2f}, "
                f"mean {float(series.mean()):.2f}, max {float(series.max()):.2f}; это сразу намекает на выбросы или очень неоднородные заказы."
            )
    missing = df.isna().sum()
    missing_columns = [str(column) for column, count in missing.items() if int(count) > 0]
    if missing_columns:
        parts.append(f"По качеству данных есть пропуски в {', '.join(missing_columns[:4])}.")
    if dimensions:
        parts.append(f"Для объяснения различий наиболее полезны сегменты: {', '.join(f'`{item}`' for item in dimensions[:5])}.")
    if timestamps:
        parts.append(f"Динамику можно проверять через `{timestamps[0]}`.")
    return " ".join(parts)


def _candidate_relationship_evidence(
    df: pd.DataFrame,
    metrics: list[str],
    dimensions: list[str],
    timestamps: list[str],
) -> list[str]:
    evidence: list[str] = []
    business_metrics = [metric for metric in metrics if not _looks_identifier_like(df[metric], metric)]
    metric = _preferred_business_metric(business_metrics) or (business_metrics[0] if business_metrics else None)
    if metric and dimensions:
        scored_dimensions: list[tuple[float, str, str]] = []
        for dimension in dimensions[:8]:
            if dimension not in df.columns or metric not in df.columns:
                continue
            grouped = (
                df.groupby(dimension, dropna=False)[metric]
                .agg(["count", "mean", "median", "min", "max"])
                .dropna(subset=["mean"])
            )
            if len(grouped) < 2:
                continue
            spread = float(grouped["mean"].max() - grouped["mean"].min())
            denominator = max(abs(float(grouped["mean"].median())), 1.0)
            score = spread / denominator
            top = grouped.sort_values("mean", ascending=False).iloc[0]
            bottom = grouped.sort_values("mean", ascending=True).iloc[0]
            scored_dimensions.append(
                (
                    score,
                    dimension,
                    f"`{metric}` visibly varies by `{dimension}`: highest average is `{top.name}` ({float(top['mean']):.2f}), "
                    f"lowest is `{bottom.name}` ({float(bottom['mean']):.2f}).",
                )
            )
        for _, _, text in sorted(scored_dimensions, reverse=True)[:2]:
            evidence.append(text)
    if len(business_metrics) >= 2:
        numeric = df[business_metrics[:8]].apply(pd.to_numeric, errors="coerce")
        corr_pairs: list[tuple[float, str]] = []
        for other in business_metrics:
            if other == metric or metric is None:
                continue
            pair = numeric[[metric, other]].dropna()
            if len(pair) < 3 or pair[metric].nunique() < 2 or pair[other].nunique() < 2:
                continue
            corr = float(pair[metric].corr(pair[other]))
            corr_pairs.append((abs(corr), f"`{metric}` and `{other}` have a {_correlation_strength(corr)} linear relationship candidate (correlation {corr:.2f})."))
        if corr_pairs:
            evidence.append(sorted(corr_pairs, reverse=True)[0][1])
    if metric and timestamps:
        evidence.append(f"`{timestamps[0]}` can test whether `{metric}` changes over time rather than only across groups.")
    return evidence


def _business_questions_response(question: str, df: pd.DataFrame) -> dict[str, Any]:
    semantic = build_semantic_dataset_profile(df=df)
    metrics = [item["name"] for item in _rank_metric_candidates(df, semantic)[:4]]
    dimensions = [column.name for column in semantic.dimensions[:5]]
    timestamps = [column.name for column in semantic.timestamps[:2]]
    questions: list[str] = []
    metric = metrics[0] if metrics else None
    if metric and dimensions:
        questions.append(f"Which `{dimensions[0]}` segments have the highest and lowest `{metric}` values?")
    if metric and len(dimensions) > 1:
        questions.append(f"Does the `{metric}` difference remain inside `{dimensions[1]}`, or is it explained by subgroup mix?")
    if metric and timestamps:
        questions.append(f"Is there a stable trend, seasonality, or sharp spike in `{metric}` over `{timestamps[0]}`?")
    if metric:
        questions.append(f"Which records look like `{metric}` outliers and change the averages?")
    if len(metrics) > 1:
        questions.append(f"Which numeric fields are most strongly related to `{metric}`: {', '.join(f'`{item}`' for item in metrics[1:4])}?")
    if not questions:
        questions.append("Which categories have the largest record volume, and where is the data imbalanced?")
    summary = (
        "This is a broad exploratory question, so I am not binding it to the previous branch. "
        "The strongest investigation questions for this dataset are: "
        + " ".join(f"{idx + 1}. {item}" for idx, item in enumerate(questions[:5]))
    )
    return _output(
        question=question,
        summary=summary,
        findings=[],
        evidence=[f"Ranked analytical questions from {len(df):,} rows, candidate metrics, dimensions, and timestamps."],
        limitations=["These are investigation directions; each needs a focused calculation before becoming a finding."],
        next_steps=questions[:5],
        code="",
        result_preview="",
        timeline=_timeline("business_question_generation", metrics=len(metrics), dimensions=len(dimensions)),
        artifacts=[],
        trace_metadata={"fallback": "business_questions", "analysis_type": "exploration", "suppress_key_findings": True},
    )


def _important_fields_response(question: str, df: pd.DataFrame) -> dict[str, Any]:
    semantic = build_semantic_dataset_profile(df=df)
    ranked_metrics = _rank_metric_candidates(df, semantic)
    dimensions = [column.name for column in semantic.dimensions[:6]]
    timestamps = [column.name for column in semantic.timestamps[:3]]
    identifiers = [column.name for column in semantic.identifiers[:6]]
    metric_text = "; ".join(f"`{item['name']}` - {item['reason']}" for item in ranked_metrics[:5]) or "no reliable default numeric metrics"
    parts = [
        f"The most important fields depend on the question, but the first analytical priority is the metric set: {metric_text}.",
    ]
    if dimensions:
        parts.append("For explaining differences, the strongest segmentation fields are: " + ", ".join(f"`{item}`" for item in dimensions[:6]) + ".")
    if timestamps:
        parts.append("For time-based analysis, use: " + ", ".join(f"`{item}`" for item in timestamps[:3]) + ".")
    if identifiers:
        parts.append("Identifiers and code-like fields are useful for joins, counts, and deduplication, but not as default average metrics: " + ", ".join(f"`{item}`" for item in identifiers[:5]) + ".")
    return _output(
        question=question,
        summary=" ".join(parts),
        findings=[],
        evidence=["Ranked fields by semantic role, numeric usefulness, cardinality, sparsity, and distribution spread."],
        limitations=["Field importance is task-dependent; this ranking is a starting analytical map, not a final conclusion."],
        next_steps=[
            "Choose one top metric and compare it across the strongest segmentation fields.",
            "Check whether identifier/code-like fields are needed only for counts, joins, or deduplication.",
        ],
        code="",
        result_preview="",
        timeline=_timeline("important_fields_scan", metrics=len(ranked_metrics), dimensions=len(dimensions)),
        artifacts=[],
        trace_metadata={"fallback": "important_fields", "analysis_type": "exploration", "suppress_key_findings": True},
    )


def _preferred_business_metric(metrics: list[str]) -> str | None:
    preferred = (
        "sales", "revenue", "profit", "amount", "price", "cost", "value", "total", "score", "rate",
        "выруч", "продаж", "прибыл", "сумм", "стоим", "оцен", "рейтинг", "доля",
    )
    for metric in metrics:
        lowered = metric.lower()
        if any(token in lowered for token in preferred):
            return metric
    return None


def _rank_metric_candidates(df: pd.DataFrame, semantic: Any | None = None) -> list[dict[str, Any]]:
    semantic = semantic or build_semantic_dataset_profile(df=df)
    semantic_by_name = getattr(semantic, "by_name", {})
    candidates: list[dict[str, Any]] = []
    for column in df.select_dtypes(include="number").columns:
        name = str(column)
        series = pd.to_numeric(df[column], errors="coerce")
        non_null = series.dropna()
        if non_null.empty:
            continue
        profile = semantic_by_name.get(name) if isinstance(semantic_by_name, dict) else None
        normalized = _normalize(name)
        unique_ratio = float(non_null.nunique(dropna=True) / max(len(non_null), 1))
        sparsity = float(series.isna().mean())
        variance = float(non_null.std()) if len(non_null) > 1 else 0.0
        mean_abs = max(abs(float(non_null.mean())), 1.0)
        relative_variance = variance / mean_abs
        score = 50.0
        penalties: list[str] = []
        bonuses: list[str] = []
        if _looks_identifier_like(df[column], name) or _contains_any(normalized, ("id", "uuid", "key", "code", "postal", "zip", "postcode", "invoice", "order number", "row")):
            score -= 70
            penalties.append("identifier/code-like")
        semantic_tags = set(getattr(profile, "semantic_tags", []) or [])
        if semantic_tags & {"money", "quality", "percentage", "ranking"}:
            score += 22
            bonuses.append("business-like numeric meaning")
        if _contains_any(normalized, ("amount", "value", "total", "score", "rate", "ratio", "price", "cost", "revenue", "sales", "profit", "quantity", "duration")):
            score += 18
            bonuses.append("aggregation-friendly name")
        if unique_ratio > 0.95 and len(non_null) > 20:
            score -= 28
            penalties.append("near-unique values")
        elif unique_ratio < 0.02 and len(non_null) > 100:
            score -= 8
            penalties.append("very low variation")
        if sparsity > 0.4:
            score -= 20
            penalties.append("sparse")
        if relative_variance > 0.05:
            score += min(18, relative_variance * 6)
            bonuses.append("useful spread")
        if non_null.nunique(dropna=True) <= 2:
            score -= 15
            penalties.append("binary/flag-like")
        if score <= 0:
            continue
        reason = ", ".join(bonuses[:2]) if bonuses else "numeric field with usable variation"
        if penalties and not bonuses:
            reason = "lower-priority numeric field; " + ", ".join(penalties[:2])
        candidates.append({"name": name, "score": score, "reason": reason, "penalties": penalties})
    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def _dependency_directions(
    metrics: list[str],
    dimensions: list[str],
    timestamps: list[str],
    *,
    primary_metric: str | None = None,
) -> list[str]:
    directions = []
    metric = primary_metric or (metrics[0] if metrics else None)
    companion_metrics = [item for item in metrics if item != metric]
    if metric and dimensions:
        directions.append(f"compare `{metric}` across `{dimensions[0]}`")
    if metric and len(dimensions) > 1:
        directions.append(f"check whether `{dimensions[1]}` explains differences in `{metric}`")
    if metric and companion_metrics:
        directions.append(f"test relationships between `{metric}` and `{companion_metrics[0]}`")
    if metric and timestamps:
        directions.append(f"track `{metric}` over time using `{timestamps[0]}`")
    if metric and dimensions:
        directions.append(f"look for unusual `{dimensions[0]}` groups with extreme `{metric}`")
    return directions


def _dataset_subject_hint(metrics: list[str], dimensions: list[str], timestamps: list[str]) -> str:
    lowered = " ".join([*metrics, *dimensions, *timestamps]).lower()
    if any(token in lowered for token in ("role", "job", "position", "profession", "salary", "applicant", "opening")):
        return "job posting or hiring-market data"
    if any(token in lowered for token in ("order", "sale", "revenue", "customer", "product", "ship")):
        return "commercial transaction or sales data"
    if any(token in lowered for token in ("rating", "score", "survey", "response")):
        return "performance, rating, or survey-style data"
    if any(token in lowered for token in ("city", "region", "country", "location")):
        return "geographic analytical data"
    return "a structured tabular dataset"


def _analytical_directions(metrics: list[str], dimensions: list[str], timestamps: list[str]) -> list[str]:
    directions = []
    metric = metrics[0] if metrics else None
    dimension = dimensions[0] if dimensions else None
    timestamp = timestamps[0] if timestamps else None
    if metric and dimension:
        directions.append(f"compare `{metric}` across `{dimension}`")
        directions.append(f"identify `{dimension}` groups with unusually high or low `{metric}`")
    if metric:
        directions.append(f"check distribution, spread, and outliers in `{metric}`")
    if metric and timestamp:
        directions.append(f"track `{metric}` over time using `{timestamp}`")
    if len(metrics) >= 2:
        directions.append(f"test relationships between `{metrics[0]}` and `{metrics[1]}`")
    if dimensions:
        directions.append(f"segment records using `{dimensions[0]}`")
    return directions


def _selection_text_for_question(question: str, data_context: dict[str, Any] | None) -> str:
    """Use recent analytical context to resolve terse follow-ups without overriding explicit entities."""

    if not isinstance(data_context, dict):
        return question
    if not (_is_chart_question(question) or _is_analytical_follow_up(question) or _is_contextual_follow_up(question)):
        return question
    context = data_context.get("conversation_context")
    if not isinstance(context, dict):
        return question
    parts = [question]
    chart_context = context.get("latest_chart_context")
    if isinstance(chart_context, dict):
        for key in ("metric", "dimension", "chart_type", "title"):
            value = str(chart_context.get(key) or "").strip()
            if value:
                parts.append(value)
    messages = context.get("messages")
    if isinstance(messages, list):
        for message in messages[-6:]:
            if isinstance(message, dict):
                parts.append(str(message.get("content") or ""))
    latest_findings = context.get("latest_findings")
    if isinstance(latest_findings, list):
        parts.extend(str(item) for item in latest_findings[:5])
    artifact_titles = context.get("artifact_titles")
    if isinstance(artifact_titles, list):
        parts.extend(str(item) for item in artifact_titles[:5])
    return " ".join(part for part in parts if part.strip())


def _missing_entity_response(question: str, df: pd.DataFrame, metric_col: str, entity: str) -> dict[str, Any]:
    semantic = build_semantic_dataset_profile(df=df)
    available_dimensions = [column.name for column in semantic.dimensions[:8]]
    nearest = available_dimensions[0] if available_dimensions else ""
    summary = (
        f"I found `{metric_col}` as the metric, but I could not find a {entity}-level field in this dataset. "
        f"I did not substitute `{nearest}` silently because the requested grouping was explicit."
        if nearest
        else f"I found `{metric_col}` as the metric, but I could not find a {entity}-level grouping field in this dataset."
    )
    if nearest:
        summary += f" The nearest available grouping field is `{nearest}`."
    return _output(
        question=question,
        summary=summary,
        findings=[],
        evidence=[f"Checked available columns: {', '.join(str(col) for col in df.columns[:12])}."],
        limitations=[f"No {entity}-like column was available for the requested chart or comparison."],
        next_steps=[f"Use `{nearest}` instead if that grouping is acceptable." if nearest else "Add a dataset field with the requested grouping and rerun the analysis."],
        code="",
        result_preview="",
        timeline=_timeline("missing_explicit_entity", entity=entity, metric=metric_col),
        artifacts=[],
        trace_metadata={"fallback": "missing_explicit_entity", "analysis_type": "fallback", "suppress_key_findings": True},
    )


def _missing_metric_response(
    question: str,
    requested_metric: str,
    available_metrics: list[str],
    *,
    source_name: str,
) -> dict[str, Any]:
    alternatives = [metric for metric in available_metrics if metric]
    suggestion = (
        " Вместо этого здесь можно построить график по "
        + ", ".join(f"`{metric}`" for metric in alternatives[:5])
        + "."
        if alternatives
        else " В доступной схеме я не вижу надежной числовой метрики для такой замены."
    )
    summary = (
        f"График по `{requested_metric}` построить нельзя: в `{source_name}` нет такого поля. "
        "Я не буду молча подменять запрошенную метрику на другую."
        + suggestion
    )
    return _output(
        question=question,
        summary=summary,
        findings=[],
        evidence=[f"Checked requested metric `{requested_metric}` against available dataset columns."],
        limitations=[f"No `{requested_metric}` field was available for the requested chart or comparison."],
        next_steps=[
            f"Ask for the chart again using one of: {', '.join(alternatives[:5])}." if alternatives else "Upload data with the requested metric and rerun the chart request."
        ],
        code="",
        result_preview="",
        timeline=_timeline("missing_explicit_metric", metric=requested_metric),
        artifacts=[],
        trace_metadata={"fallback": "missing_explicit_metric", "analysis_type": "fallback", "suppress_key_findings": True},
    )
def _grouped_metric_response(
    question: str,
    df: pd.DataFrame,
    metric_col: str,
    dimension_col: str,
    *,
    chart_requested: bool = False,
) -> dict[str, Any]:
    language = ResponseLanguagePolicy.from_message(question, protected_terms=df.columns)
    branch_switch = _is_branch_switch_to_dimension(question, dimension_col)
    aggregation = _grouped_metric_aggregation(question)
    value_col = "mean" if aggregation == "mean" else "median" if aggregation == "median" else "sum"
    value_label = "average" if aggregation == "mean" else "median" if aggregation == "median" else "total"
    grouped = (
        df.groupby(dimension_col, dropna=False)[metric_col]
        .agg(["count", "mean", "median", "min", "max", "std", "sum"])
        .sort_values(value_col, ascending=False)
        .reset_index()
    )
    grouped["std"] = grouped["std"].fillna(0.0)
    total = float(grouped["sum"].sum())
    grouped["share_of_total"] = grouped["sum"] / total if total else 0.0

    top = grouped.iloc[0]
    bottom = grouped.iloc[-1]
    widest = grouped.assign(spread=grouped["max"] - grouped["min"]).sort_values("spread", ascending=False).iloc[0]
    top_groups = _format_group_values(grouped.head(3), dimension_col, value_col)
    bottom_groups = _format_group_values(grouped.tail(3).sort_values(value_col), dimension_col, value_col)
    reliability_note = _group_reliability_note(grouped, dimension_col)
    reliability_note_ru = _group_reliability_note_ru(grouped, dimension_col)
    result_preview = grouped.to_string(index=False)
    top_chart = grouped.head(20)
    if language.is_russian:
        finding = (
            f"По `{dimension_col}` {value_label} `{metric_col}` выше всего у `{top[dimension_col]}` ({top[value_col]:.2f}) "
            f"и ниже всего у `{bottom[dimension_col]}` ({bottom[value_col]:.2f}). "
            f"Самый широкий внутренний разброс у `{widest[dimension_col]}` ({float(widest['max'] - widest['min']):.2f}); "
            "это может указывать на смешение подгрупп или нестабильность внутри сегмента."
        )
        summary = (
            f"`{metric_col}` по `{dimension_col}` лидирует у {top_groups}. "
            f"`{metric_col}` заметно различается по `{dimension_col}`. "
            f"Самые низкие значения: {bottom_groups}. "
            f"Самый широкий диапазон значений у `{widest[dimension_col]}` "
            f"({widest['min']:.2f}–{widest['max']:.2f}), поэтому группу нельзя читать как однородную. "
            f"Разрыв может отражать реальные различия сегментов, смешение подгрупп, выбросы или качество данных. {reliability_note_ru}"
        )
        evidence = [f"Сгруппировал `{metric_col}` по `{dimension_col}` на {len(df)} строках."]
        limitations = [
            "Это описательная агрегация по текущему датасету; она не доказывает причинность.",
            reliability_note_ru,
            f"`{widest[dimension_col]}` нужно проверить отдельно: широкий разброс может отражать выбросы, смешение подгрупп или реальную нестабильность.",
        ]
        next_steps = [
            f"Проверить, остается ли `{widest[dimension_col]}` нестабильным после удаления extreme rows.",
            f"Разбить `{metric_col}` по другому релевантному полю и проверить, сохраняется ли разрыв по `{dimension_col}`.",
            f"Использовать grouped bar chart как поддержку различий по `{dimension_col}` в отчете.",
        ]
        chart_title = average_chart_title(metric_col, dimension_col, language) if aggregation == "mean" else f"{metric_col} by {dimension_col}"
        ranking_scope = f"Top {min(len(top_chart), 20)} групп `{dimension_col}` по {value_label} `{metric_col}` из {len(grouped)} групп."
    else:
        finding = (
            f"Across `{dimension_col}`, {value_label} `{metric_col}` is highest in `{top[dimension_col]}` ({top[value_col]:.2f}) "
            f"and lowest in `{bottom[dimension_col]}` ({bottom[value_col]:.2f}). "
            f"`{widest[dimension_col]}` has the widest internal spread ({float(widest['max'] - widest['min']):.2f}), "
            "which may indicate hidden subgroup mix or less stable behavior inside that segment."
        )
        summary = (
            f"`{metric_col}` by `{dimension_col}` is led by {top_groups} using {value_label} `{metric_col}`. "
            f"`{metric_col}` differs sharply across `{dimension_col}`. "
            f"Lowest values: {bottom_groups}. "
            f"The widest value band is `{widest[dimension_col]}` "
            f"({widest['min']:.2f} to {widest['max']:.2f}), which suggests the group is not internally uniform. "
            f"The gap may reflect real segment differences, hidden subgroup mix, outliers, or data quality effects. {reliability_note}"
        )
        if branch_switch:
            transition = BranchTransitionSynthesizer.synthesize(
                metric=metric_col,
                dimension=dimension_col,
                leaders=[
                    {dimension_col: row[dimension_col], "mean": row["mean"], "count": row["count"]}
                    for _, row in top_chart.head(3).iterrows()
                ],
                caveat=(
                    f"Read it carefully: {reliability_note} "
                    f"`{widest[dimension_col]}` has the widest spread ({widest['min']:.2f} to {widest['max']:.2f}), so outlier and sample-size checks still matter."
                ),
            )
            summary = (
                transition
            )
        evidence = [f"Grouped `{metric_col}` by `{dimension_col}` over {len(df)} rows."]
        limitations = [
            "This is descriptive aggregation over the current dataset; it does not prove causality.",
            reliability_note,
            f"`{widest[dimension_col]}` needs validation because wide spread can reflect outliers, mixed subgroups, or true instability.",
        ]
        next_steps = [
            f"Check whether `{widest[dimension_col]}` remains unusually variable after removing extreme rows.",
            f"Break down `{metric_col}` by another relevant field to see whether the `{dimension_col}` gap persists.",
            f"Use the grouped bar chart to support the strongest `{dimension_col}` differences in a report.",
        ]
        chart_title = average_chart_title(metric_col, dimension_col, language).replace("`", "") if aggregation == "mean" else f"{metric_col} by {dimension_col}"
        ranking_scope = f"Top {min(len(top_chart), 20)} `{dimension_col}` groups by {value_label} `{metric_col}` from {len(grouped)} total groups."
    code = (
        f"result = (df.groupby({dimension_col!r}, dropna=False)[{metric_col!r}]\n"
        "    .agg(['count', 'mean', 'median', 'min', 'max', 'std', 'sum'])\n"
        f"    .sort_values({value_col!r}, ascending=False)\n"
        "    .reset_index())\n"
        "result['share_of_total'] = result['sum'] / result['sum'].sum() if result['sum'].sum() else 0.0"
    )
    timeline = _timeline("deterministic_pandas_fallback", metric_col=metric_col, dimension_col=dimension_col, rows=len(df))
    chart_artifact = {
        "artifact_type": "chart",
        "title": chart_title,
        "content": {
            "chart_type": "bar",
            "x": dimension_col,
            "y": value_col,
            "metric": metric_col,
            "dimension": dimension_col,
            "aggregation": aggregation,
            "filters": [],
            "rows": top_chart[[dimension_col, "sum", "mean", "median", "count"]].to_dict(orient="records"),
            "ranking_scope": ranking_scope,
            "row_count": int(len(df)),
        },
        "visibility": "user",
        "pinned": True,
        "metadata": {
            "metric": metric_col,
            "dimension": dimension_col,
            "aggregation": aggregation,
            "ranking_mode": "descending",
            "chart_type": "bar",
            "filters": [],
            "top_n": int(min(len(top_chart), 20)),
            "sorted_values": top_chart[[dimension_col, value_col, "count"]].to_dict(orient="records"),
            "transformation_state": "raw",
            "ranking_scope": ranking_scope,
            "row_count": int(len(df)),
            "recommended": True,
        },
    }

    return _output(
        question=question,
        summary=summary,
        findings=[finding],
        evidence=evidence,
        limitations=limitations,
        next_steps=next_steps,
        code=code,
        result_preview=result_preview,
        timeline=timeline,
        artifacts=[
            {
                "artifact_type": "table",
                "title": f"{metric_col} by {dimension_col}",
                "content": grouped.to_dict(orient="records"),
                "visibility": "user",
                "pinned": True,
                "metadata": {"preview": result_preview},
            },
            chart_artifact,
        ],
        trace_metadata={
            "fallback": "deterministic_pandas",
            "metric": metric_col,
            "dimension": dimension_col,
            "chart_requested": chart_requested,
        },
    )


def _group_count_relationship_response(
    question: str,
    df: pd.DataFrame,
    metric_col: str,
    dimension_col: str,
) -> dict[str, Any]:
    working = df[[dimension_col, metric_col]].copy()
    working[metric_col] = pd.to_numeric(working[metric_col], errors="coerce")
    working = working.dropna(subset=[metric_col])
    if working.empty:
        return _grouped_metric_response(question, df, metric_col, dimension_col)

    grouped = (
        working.groupby(dimension_col, dropna=False)[metric_col]
        .agg(["count", "sum", "mean", "median", "min", "max"])
        .reset_index()
        .rename(columns={"count": "record_count", "sum": "metric_total"})
    )
    grouped["avg_value"] = grouped["mean"]
    grouped["count_share"] = grouped["record_count"] / max(float(grouped["record_count"].sum()), 1.0)
    grouped["metric_share"] = grouped["metric_total"] / max(float(grouped["metric_total"].sum()), 1.0)
    grouped["value_vs_volume_gap"] = grouped["metric_share"] - grouped["count_share"]
    by_total = grouped.sort_values(["metric_total", "record_count"], ascending=[False, False])
    by_gap = grouped.assign(abs_gap=lambda data: data["value_vs_volume_gap"].abs()).sort_values(
        ["abs_gap", "metric_total"],
        ascending=[False, False],
    )
    leader = by_total.iloc[0]
    gap_leader = by_gap.iloc[0]
    correlation = 0.0
    if len(grouped) >= 2 and grouped["record_count"].nunique(dropna=True) > 1 and grouped["metric_total"].nunique(dropna=True) > 1:
        correlation = float(grouped["record_count"].corr(grouped["metric_total"]))
    strength = _correlation_strength(correlation)
    top_groups = _format_group_values(by_total.head(3), dimension_col, "metric_total")
    by_avg = grouped.sort_values(["avg_value", "record_count"], ascending=[False, False])
    avg_groups = _format_group_values(by_avg.head(3), dimension_col, "avg_value")
    high_volume_modest_avg = grouped.sort_values("record_count", ascending=False).head(10).sort_values("avg_value").head(1).iloc[0]
    high_avg_low_volume = by_avg.head(10).sort_values("record_count").head(1).iloc[0]
    direction = "closely tracks" if abs(correlation) >= 0.7 else "partly tracks" if abs(correlation) >= 0.35 else "does not strongly track"
    summary = (
        f"`{metric_col}` {direction} record volume across `{dimension_col}` groups. "
        f"By total `{metric_col}`, the largest groups are {top_groups}. "
        f"`{leader[dimension_col]}` leads with total `{metric_col}` of {float(leader['metric_total']):.2f} "
        f"across {int(leader['record_count'])} records. "
        f"The record-count relationship is {strength} (correlation {correlation:.2f}), so volume explains "
        f"{'much of' if abs(correlation) >= 0.7 else 'part of' if abs(correlation) >= 0.35 else 'only a limited share of'} "
        f"the total-value gap. By average `{metric_col}` per record, the strongest groups are {avg_groups}. "
        f"`{high_avg_low_volume[dimension_col]}` is high on average but low on volume ({float(high_avg_low_volume['avg_value']):.2f} avg across {int(high_avg_low_volume['record_count'])} records), "
        f"while `{high_volume_modest_avg[dimension_col]}` has high volume but weaker average value ({float(high_volume_modest_avg['avg_value']):.2f} avg across {int(high_volume_modest_avg['record_count'])} records). "
        f"`{gap_leader[dimension_col]}` has the largest value-versus-volume imbalance, so the interpretation should separate volume, average order value, and outlier influence."
    )
    result_rows = by_total.head(25).to_dict(orient="records")
    scatter_rows = grouped[[dimension_col, "record_count", "metric_total", "avg_value", "value_vs_volume_gap"]].to_dict(orient="records")
    timeline = _timeline(
        "thread_count_relationship_check",
        metric_col=metric_col,
        dimension_col=dimension_col,
        groups=len(grouped),
        rows=len(working),
    )
    return _output(
        question=question,
        summary=summary,
        findings=[
            f"`{leader[dimension_col]}` is strongest by total `{metric_col}` with {float(leader['metric_total']):.2f} across {int(leader['record_count'])} records.",
            f"Record volume has a {strength} association with total `{metric_col}` across `{dimension_col}` groups (correlation {correlation:.2f}).",
            f"`{gap_leader[dimension_col]}` has the largest imbalance between `{metric_col}` share and record-count share, which deserves validation.",
        ],
        evidence=[
            f"Compared total `{metric_col}`, average `{metric_col}`, and record counts by `{dimension_col}` over {len(working):,} valid rows.",
            "Used group-level correlation between record count and total metric value as volume evidence.",
        ],
        limitations=[
            "Record count can explain total value without explaining average value.",
            "A high-volume group may still be weak on average; compare both totals and averages before making a decision.",
            "Sparse groups can look unusually strong or weak because one record can move the result.",
        ],
        next_steps=[
            f"Compare average `{metric_col}` and total `{metric_col}` side by side for the top `{dimension_col}` groups.",
            f"Check whether the high-total `{dimension_col}` groups also have unusually high average `{metric_col}`.",
            "Inspect the largest value-versus-volume imbalance for possible mix effects or outliers.",
        ],
        code=(
            f"result = df.groupby({dimension_col!r})[{metric_col!r}]\n"
            "    .agg(record_count='count', metric_total='sum', avg_value='mean', median='median')"
        ),
        result_preview=pd.DataFrame(result_rows).to_string(index=False),
        timeline=timeline,
        artifacts=[
            {
                "artifact_type": "table",
                "title": f"{metric_col} and record volume by {dimension_col}",
                "content": result_rows,
                "visibility": "user",
                "pinned": True,
                "metadata": {"metric": metric_col, "dimension": dimension_col, "analysis_type": "volume_relationship"},
            },
            {
                "artifact_type": "chart",
                "title": f"{metric_col} vs record volume by {dimension_col}",
                "content": {
                    "chart_type": "scatter",
                    "x": "record_count",
                    "y": "metric_total",
                    "metric": metric_col,
                    "dimension": dimension_col,
                    "rows": scatter_rows,
                },
                "visibility": "user",
                "pinned": True,
                "metadata": {"metric": metric_col, "dimension": dimension_col, "analysis_type": "volume_relationship"},
            },
        ],
        trace_metadata={
            "fallback": "thread_count_relationship_check",
            "metric": metric_col,
            "dimension": dimension_col,
            "correlation": correlation,
        },
    )


def _group_concentration_hypothesis_response(
    question: str,
    df: pd.DataFrame,
    metric_col: str,
    dimension_col: str,
) -> dict[str, Any]:
    working = df[[dimension_col, metric_col]].copy()
    working[metric_col] = pd.to_numeric(working[metric_col], errors="coerce")
    working = working.dropna(subset=[metric_col])
    if working.empty:
        return _metric_coverage_response(question, df)
    rows = []
    for value, group in working.groupby(dimension_col, dropna=False):
        total = float(group[metric_col].sum())
        top_value = float(group[metric_col].max()) if len(group) else 0.0
        top_share = top_value / total if total else 0.0
        rows.append(
            {
                dimension_col: value,
                "count": int(len(group)),
                "total": total,
                "mean": float(group[metric_col].mean()),
                "median": float(group[metric_col].median()),
                "top_order_value": top_value,
                "top_order_share": top_share,
                "mean_median_gap": float(group[metric_col].mean() - group[metric_col].median()),
            }
        )
    result = pd.DataFrame(rows).sort_values(["total", "top_order_share"], ascending=[False, False])
    top = result.head(5)
    concentrated = top[top["top_order_share"] >= 0.5]
    if not concentrated.empty:
        concentration_text = ", ".join(
            f"`{row[dimension_col]}` ({row['top_order_share']:.1%} from top order, n={int(row['count'])})"
            for _, row in concentrated.head(3).iterrows()
        )
        conclusion = f"The hypothesis is plausible for {concentration_text}."
    else:
        conclusion = "The top city ranking is not obviously driven by a single large order in the leading groups."
    summary = (
        f"Hypothesis check for `{metric_col}` by `{dimension_col}`: {conclusion} "
        f"This is a top-record concentration test: I compared total `{metric_col}`, average vs median, record count, and the largest-record share for each `{dimension_col}` group. "
        f"If a `{dimension_col}` group has high total or average `{metric_col}` but a very high top-record share, the ranking is fragile."
    )
    result_rows = result.head(25).to_dict(orient="records")
    return _output(
        question=question,
        summary=summary,
        findings=[summary],
        evidence=[f"Computed top-order concentration for `{metric_col}` across `{dimension_col}`."],
        limitations=["Rows are treated as order lines unless a separate order-level aggregation is available."],
        next_steps=[f"Re-rank `{dimension_col}` after removing the largest `{metric_col}` row per group.", "Compare city median ranking against total and mean ranking."],
        code="grouped = df.groupby(dimension)[metric].agg(['count', 'sum', 'mean', 'median', 'max'])",
        result_preview=result.head(15).to_string(index=False),
        timeline=_timeline("hypothesis_concentration_check", metric_col=metric_col, dimension_col=dimension_col, rows=len(df)),
        artifacts=[
            {
                "artifact_type": "table",
                "title": f"{metric_col} concentration by {dimension_col}",
                "content": result_rows,
                "visibility": "user",
                "pinned": True,
                "metadata": {"branch_type": "hypothesis_validation", "metric": metric_col, "dimension": dimension_col},
            }
        ],
        trace_metadata={"fallback": "hypothesis_concentration_check", "analysis_type": "hypothesis_validation", "active_branch_type": "hypothesis_validation", "metric": metric_col, "dimension": dimension_col},
    )


def _category_value_volume_hypothesis_response(
    question: str,
    df: pd.DataFrame,
    metric_col: str,
    dimension_col: str,
    matched_value: str,
) -> dict[str, Any]:
    working = df[[dimension_col, metric_col]].copy()
    working[metric_col] = pd.to_numeric(working[metric_col], errors="coerce")
    working = working.dropna(subset=[metric_col])
    grouped = (
        working.groupby(dimension_col, dropna=False)[metric_col]
        .agg(count="count", total="sum", mean="mean", median="median")
        .reset_index()
    )
    grouped["share_of_rows"] = grouped["count"] / max(float(grouped["count"].sum()), 1.0)
    grouped["share_of_total"] = grouped["total"] / max(float(grouped["total"].sum()), 1.0)
    grouped = grouped.sort_values("total", ascending=False)
    target_rows = grouped[grouped[dimension_col].astype(str).str.lower() == str(matched_value).lower()]
    target = target_rows.iloc[0] if not target_rows.empty else None
    total_leader = grouped.iloc[0]
    mean_leader = grouped.sort_values("mean", ascending=False).iloc[0]
    if target is not None:
        volume_driven = bool(target[dimension_col] == total_leader[dimension_col] and target[dimension_col] != mean_leader[dimension_col])
        conclusion = (
            f"`{matched_value}` leads total `{metric_col}` mostly through volume: it has {int(target['count'])} records "
            f"({float(target['share_of_rows']):.1%} of rows) and {float(target['share_of_total']):.1%} of total `{metric_col}`, "
            f"while the highest average is `{mean_leader[dimension_col]}` ({float(mean_leader['mean']):.2f})."
            if volume_driven
            else f"`{matched_value}` is not purely volume-driven from this check: total rank, average rank, and record share need to be read together."
        )
    else:
        conclusion = f"I matched `{matched_value}` to `{dimension_col}`, but it was not present after filtering valid `{metric_col}` rows."
    summary = (
        f"Hypothesis check for `{matched_value}` in `{dimension_col}` using `{metric_col}`: {conclusion} "
        f"This tests dominance by comparing total `{metric_col}` against average `{metric_col}` and row volume."
    )
    rows = grouped.to_dict(orient="records")
    return _output(
        question=question,
        summary=summary,
        findings=[summary],
        evidence=[f"Matched `{matched_value}` as a category value of `{dimension_col}` and compared total, mean, median, and count."],
        limitations=["Row count is a proxy for order volume; use order-level counts if one order can span multiple rows."],
        next_steps=[f"Repeat with distinct order counts by `{dimension_col}` if an order identifier is available."],
        code="df.groupby(dimension)[metric].agg(count='count', total='sum', mean='mean', median='median')",
        result_preview=grouped.to_string(index=False),
        timeline=_timeline("hypothesis_volume_check", metric_col=metric_col, dimension_col=dimension_col, rows=len(df)),
        artifacts=[
            {
                "artifact_type": "table",
                "title": f"{metric_col} volume vs value by {dimension_col}",
                "content": rows,
                "visibility": "user",
                "pinned": True,
                "metadata": {"branch_type": "hypothesis_validation", "metric": metric_col, "dimension": dimension_col, "matched_value": matched_value},
            }
        ],
        trace_metadata={"fallback": "hypothesis_volume_check", "analysis_type": "hypothesis_validation", "active_branch_type": "hypothesis_validation", "metric": metric_col, "dimension": dimension_col, "matched_category_value": matched_value},
    )


def _group_unusual_values_response(
    question: str,
    df: pd.DataFrame,
    metric_col: str,
    dimension_col: str,
) -> dict[str, Any]:
    working = df[[dimension_col, metric_col]].copy()
    working[metric_col] = pd.to_numeric(working[metric_col], errors="coerce")
    working = working.dropna(subset=[metric_col])
    if working.empty:
        return _outlier_response(question, df, metric_col)

    grouped = (
        working.groupby(dimension_col, dropna=False)[metric_col]
        .agg(["count", "mean", "median", "min", "max", "std"])
        .reset_index()
    )
    grouped["std"] = grouped["std"].fillna(0.0)
    grouped["spread"] = grouped["max"] - grouped["min"]

    series = working[metric_col]
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_mask = series.lt(lower) | series.gt(upper)
    flagged_total = int(outlier_mask.sum())
    outlier_counts = (
        working.loc[outlier_mask]
        .groupby(dimension_col, dropna=False)[metric_col]
        .size()
        .rename("outlier_count")
        .reset_index()
    )
    grouped = grouped.merge(outlier_counts, on=dimension_col, how="left")
    grouped["outlier_count"] = grouped["outlier_count"].fillna(0).astype(int)
    grouped["low_sample"] = grouped["count"] < 3

    by_mean = grouped.sort_values(["mean", "count"], ascending=[False, False])
    by_spread = grouped.sort_values(["spread", "std", "count"], ascending=[False, False, False])
    by_outliers = grouped.sort_values(["outlier_count", "spread", "mean"], ascending=[False, False, False])
    top = by_mean.iloc[0]
    bottom = by_mean.iloc[-1]
    widest = by_spread.iloc[0]
    outlier_leader = by_outliers.iloc[0]
    widest_groups = _format_group_values(by_spread.head(3), dimension_col, "spread")
    reliability_note = _group_reliability_note(grouped, dimension_col)
    outlier_intensity = "value-driven"
    if int(outlier_leader["outlier_count"]) > 0 and int(outlier_leader["count"]) >= max(5, int(grouped["count"].median())):
        outlier_intensity = "both value- and volume-driven"

    summary = (
        f"I checked anomalies inside the active `{metric_col}` by `{dimension_col}` comparison using IQR bounds ({lower:.2f} to {upper:.2f}), group spread, and group sample size. "
        f"{flagged_total} rows are flagged as candidate `{metric_col}` outliers. "
        f"Unusual `{metric_col}` values are concentrated in the most variable `{dimension_col}` groups, so the pattern is mainly {outlier_intensity}. "
        f"`{widest[dimension_col]}` has the widest spread ({widest['min']:.2f} to {widest['max']:.2f}); "
        f"the widest groups are {widest_groups}. "
        f"`{top[dimension_col]}` also leads by average `{metric_col}` ({top['mean']:.2f}), "
        f"while `{bottom[dimension_col]}` is lowest ({bottom['mean']:.2f}). "
        f"{reliability_note}"
    )
    if int(outlier_leader["outlier_count"]):
        summary += (
            f" Possible outliers are most concentrated in `{outlier_leader[dimension_col]}` "
            f"({int(outlier_leader['outlier_count'])} flagged value"
            f"{'' if int(outlier_leader['outlier_count']) == 1 else 's'} across {int(outlier_leader['count'])} rows)."
        )

    result = grouped.sort_values(["outlier_count", "spread", "mean"], ascending=[False, False, False])
    result_rows = result.head(25).to_dict(orient="records")
    low_sample_groups = [str(row[dimension_col]) for _, row in result.iterrows() if bool(row["low_sample"])][:5]
    findings = [
        f"`{top[dimension_col]}` leads by average `{metric_col}` at {top['mean']:.2f} across {int(top['count'])} rows; the evidence is stronger when this group has enough observations.",
        f"`{widest[dimension_col]}` is the least stable group by `{metric_col}` spread ({widest['spread']:.2f}), which could reflect outliers or multiple subsegments.",
    ]
    if int(outlier_leader["outlier_count"]):
        findings.append(
            f"`{outlier_leader[dimension_col]}` contains the most IQR outlier candidates "
            f"({int(outlier_leader['outlier_count'])})."
        )
    if low_sample_groups:
        findings.append(
            "Small-sample groups need caution: " + ", ".join(f"`{item}`" for item in low_sample_groups) + "."
        )
    findings.append(reliability_note)

    chart_rows = by_mean.head(20)[[dimension_col, "mean", "count", "std", "outlier_count"]].to_dict(orient="records")
    timeline = _timeline(
        "group_unusual_values_check",
        metric_col=metric_col,
        dimension_col=dimension_col,
        groups=len(grouped),
        rows=len(working),
    )
    return _output(
        question=question,
        summary=summary,
        findings=findings,
        evidence=[
            f"Grouped `{metric_col}` by `{dimension_col}` over {len(working):,} valid rows.",
            f"Flagged potential outliers with the IQR rule outside {lower:.2f} to {upper:.2f}.",
        ],
        limitations=[
            "Groups with very small sample sizes can look unusual because of one or two observations.",
            "Statistical outliers should be reviewed with domain context before treating them as errors.",
            "Wide dispersion can indicate real instability, mixed populations, or data quality issues.",
        ],
        next_steps=[
            f"Check the high-spread `{dimension_col}` groups against their row counts and extreme records.",
            f"Compare mean and median `{metric_col}` to separate stable differences from outlier-driven effects.",
            f"Use the chart artifact to show where unusual `{metric_col}` values cluster.",
        ],
        code=(
            f"result = df.groupby({dimension_col!r})[{metric_col!r}]\n"
            "    .agg(['count', 'mean', 'median', 'min', 'max', 'std'])\n"
            "    .assign(spread=lambda data: data['max'] - data['min'])"
        ),
        result_preview=pd.DataFrame(result_rows).to_string(index=False),
        timeline=timeline,
        artifacts=[
            {
                "artifact_type": "table",
                "title": f"Unusual {metric_col} groups by {dimension_col}",
                "content": result_rows,
                "visibility": "user",
                "pinned": True,
                "metadata": {"metric": metric_col, "dimension": dimension_col, "lower_bound": lower, "upper_bound": upper},
            },
            {
                "artifact_type": "chart",
                "title": f"Average {metric_col} by {dimension_col}",
                "content": {
                    "chart_type": "bar",
                    "x": dimension_col,
                    "y": "mean",
                    "metric": metric_col,
                    "rows": chart_rows,
                    "ranking_scope": f"Top {min(len(chart_rows), 20)} `{dimension_col}` groups by average `{metric_col}` after anomaly scoring.",
                    "row_count": int(len(working)),
                },
                "visibility": "user",
                "pinned": True,
                "metadata": {
                    "metric": metric_col,
                    "dimension": dimension_col,
                    "aggregation": "mean",
                    "ranking_scope": f"Top {min(len(chart_rows), 20)} of {len(grouped)} groups by average {metric_col}; anomaly table ranks by outlier count and spread.",
                    "row_count": int(len(working)),
                    "recommended": True,
                },
            },
        ],
        trace_metadata={
            "fallback": "group_unusual_values_check",
            "metric": metric_col,
            "dimension": dimension_col,
            "groups": len(grouped),
        },
    )
def _column_summaries_from_context(context: dict[str, Any]) -> list[dict[str, Any]]:
    summaries = context.get("column_summaries")
    if not isinstance(summaries, list):
        return []
    out: list[dict[str, Any]] = []
    for item in summaries:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        role = item.get("inferred_role")
        if hasattr(role, "value"):
            role = role.value
        out.append(
            {
                "name": name,
                "dtype": str(item.get("dtype") or ""),
                "role": str(role or "unknown"),
                "nullable": bool(item.get("nullable", False)),
                "unique_count": item.get("unique_count"),
            }
        )
    return out


def _format_group_values(rows: pd.DataFrame, group_col: str, value_col: str) -> str:
    parts = []
    for _, row in rows.iterrows():
        parts.append(f"`{row[group_col]}` ({float(row[value_col]):.2f})")
    return ", ".join(parts) if parts else "no groups"


def _grouped_metric_aggregation(question: str) -> str:
    normalized = _normalize(question)
    if any(marker in normalized for marker in ("average", "mean", "avg", "typical", "средн")):
        return "mean"
    if any(marker in normalized for marker in ("median", "медиан")):
        return "median"
    if any(marker in normalized for marker in ("total", "summed", "sum ", "суммар", "всего", "по сумме")):
        return "sum"
    if any(marker in normalized for marker in ("revenue", "sales", "выручк", "доход", "продаж", "оборот", "принос")):
        return "sum"
    if any(marker in normalized for marker in ("top", "leading", "strongest", "biggest", "largest", "топ", "лидер", "сильн")):
        return "sum"
    return "mean"


def _group_reliability_note(grouped: pd.DataFrame, group_col: str) -> str:
    if grouped.empty or "count" not in grouped.columns:
        return "Evidence strength is limited until group sample sizes are checked."
    min_count = int(grouped["count"].min())
    median_count = float(grouped["count"].median())
    sparse_groups = grouped[grouped["count"] < 3]
    if not sparse_groups.empty:
        names = ", ".join(f"`{row[group_col]}`" for _, row in sparse_groups.head(3).iterrows())
        return (
            f"Evidence is weaker for sparse groups such as {names}; "
            f"the smallest group has {min_count} row{'' if min_count == 1 else 's'}."
        )
    if median_count < 10:
        return (
            f"Evidence is moderate because typical group size is {median_count:.0f} rows; "
            "small changes may shift the ranking."
        )
    return (
        f"Evidence is stronger for the larger groups because the median group size is {median_count:.0f} rows."
    )


def _group_reliability_note_ru(grouped: pd.DataFrame, group_col: str) -> str:
    if grouped.empty or "count" not in grouped.columns:
        return "Надежность вывода ограничена, пока не проверены размеры групп."
    min_count = int(grouped["count"].min())
    median_count = float(grouped["count"].median())
    sparse_groups = grouped[grouped["count"] < 3]
    if not sparse_groups.empty:
        names = ", ".join(f"`{row[group_col]}`" for _, row in sparse_groups.head(3).iterrows())
        return f"Доказательность слабее для малых групп, например {names}; минимальный размер группы: {min_count}."
    if median_count < 10:
        return f"Доказательность умеренная: типичный размер группы {median_count:.0f} строк, поэтому небольшие изменения могут сдвинуть рейтинг."
    return f"Доказательность выше для крупных групп: медианный размер группы {median_count:.0f} строк."


























































def _profile_based_analytical_response(
    question: str,
    conversation_context: Any,
    data_source_context: dict[str, Any],
) -> dict[str, Any]:
    columns = _column_summaries_from_context(data_source_context)
    metrics = [item["name"] for item in columns if item["role"] == "metric"]
    dimensions = [item["name"] for item in columns if item["role"] == "dimension"]
    timestamps = [item["name"] for item in columns if item["role"] == "timestamp"]
    name = str(data_source_context.get("name") or "dataset")
    missing_metric = _missing_explicit_metric_request(question, [item["name"] for item in columns])
    if missing_metric:
        return _missing_metric_response(question, missing_metric, metrics, source_name=name)
    mentioned_metrics = _mentioned_columns(metrics, question)
    mentioned_dimensions = _mentioned_columns(dimensions, question)
    semantic_metric = _semantic_column_match(
        metrics,
        question,
        {
            "metric": ("metric", "value", "amount", "score", "rating", "measure"),
            "money": (
                "revenue", "sales", "sale", "salary", "price", "cost", "profit", "income", "pay", "compensation", "spend",
                "продаж", "продажи", "выруч", "доход", "зарплат", "цена", "прибыл"
            ),
            "volume": ("count", "quantity", "volume", "duration", "openings", "applicants", "users", "records", "колич", "число"),
        },
    )
    semantic_dimension = _semantic_column_match(
        dimensions,
        question,
        _dimension_semantic_groups(),
    )
    metric = mentioned_metrics[0] if mentioned_metrics else (semantic_metric or (metrics[0] if metrics else None))
    dimension = mentioned_dimensions[0] if mentioned_dimensions else (semantic_dimension or (dimensions[0] if dimensions else None))
    timestamp = (_mentioned_columns(timestamps, question) or timestamps or [None])[0]
    if _is_outlier_question(question) and metric and dimension:
        summary = (
            f"`{dimension}` is the key lens for unusual `{metric}` values. "
            f"The highest-risk groups are likely the ones with wide dispersion, high maximum `{metric}`, or small samples with extreme observations."
        )
        findings = [
            f"`{dimension}` is the strongest available segmentation field for this anomaly question.",
            f"`{metric}` is the right measure for ranking unusual group behavior.",
        ]
        next_steps = [
            f"Rank `{dimension}` groups by median `{metric}`, max `{metric}`, and spread once the raw rows are available.",
            f"Create a grouped `{metric}` chart by `{dimension}` as evidence.",
            "Treat small-sample groups cautiously because one extreme row can distort the conclusion.",
        ]
    elif _is_chart_question(question) and metric:
        if _is_distribution_question(question):
            summary = f"A histogram of `{metric}` gives the clearest view of distribution shape, skew, and unusual values."
            findings = [f"`{metric}` can be reviewed for concentration, long tails, and outliers."]
            next_steps = [f"Plot the distribution of `{metric}`.", "Compare mean vs median to check skew."]
        elif dimension and (_is_ranking_or_group_comparison_question(question) or _is_composition_question(question)):
            summary = f"A ranked bar chart comparing average `{metric}` across `{dimension}` best matches this question."
            findings = [f"`{dimension}` provides the grouping and `{metric}` provides the comparison metric."]
            next_steps = [f"Sort `{dimension}` groups by average `{metric}`.", "Inspect top and bottom groups for sample-size effects."]
        elif _is_trend_question(question) and timestamp:
            summary = f"A line chart of `{metric}` over `{timestamp}` is the clearest visualization for this question."
            findings = [f"`{timestamp}` can structure the trend, while `{metric}` provides the measure to track."]
            next_steps = [f"Plot average `{metric}` by time period.", "Look for spikes, drops, and sustained changes."]
        elif dimension:
            summary = f"A bar chart comparing average `{metric}` across `{dimension}` is the clearest visualization for this question."
            findings = [f"`{dimension}` provides the grouping and `{metric}` provides the comparison metric."]
            next_steps = [f"Sort `{dimension}` groups by average `{metric}`.", "Inspect top and bottom groups for sample-size effects."]
        elif timestamp:
            summary = f"A line chart of `{metric}` over `{timestamp}` is the clearest visualization for this question."
            findings = [f"`{timestamp}` can structure the trend, while `{metric}` provides the measure to track."]
            next_steps = [f"Plot average `{metric}` by time period.", "Look for spikes, drops, and sustained changes."]
        else:
            summary = f"A histogram of `{metric}` gives the clearest first view of the metric distribution."
            findings = [f"`{metric}` can be reviewed for distribution shape, skew, and outliers."]
            next_steps = [f"Plot the distribution of `{metric}`.", "Compare mean vs median to check skew."]
    elif _is_trend_question(question) and metric and timestamp:
        summary = f"`{timestamp}` enables trend analysis for `{metric}`. Track average `{metric}` over time and flag periods with sharp movement."
        findings = [f"`{metric}` is the measure to monitor; `{timestamp}` is the time axis."]
        next_steps = [f"Aggregate `{metric}` by week or month using `{timestamp}`.", "Investigate the largest period-to-period changes."]
    elif _is_correlation_question(question) and metric:
        other_metrics = [item for item in metrics if item != metric]
        summary = (
            f"`{metric}` can be tested against {', '.join(other_metrics[:4]) or 'other numeric fields'} to identify the strongest relationships."
        )
        findings = [f"`{metric}` has enough numeric companions for relationship analysis and scatter evidence."]
        next_steps = [f"Compute correlations against `{metric}`.", "Plot the strongest relationship and check whether outliers drive it."]
    elif _is_quality_question(question):
        summary = f"Start with missing values, duplicate rows, and unusual numeric values in `{name}` before trusting conclusions."
        findings = ["Data quality checks should happen before interpreting rankings, trends, or outliers."]
        next_steps = ["Review missingness by column.", "Check duplicate rows.", "Run outlier checks on important numeric fields."]
    elif metric and dimension:
        summary = (
            f"I can resolve `{metric}` and `{dimension}`, but exact `{metric}` by `{dimension}` results require the raw uploaded rows. "
            "Only the saved dataset profile is available in this context, so I cannot compute the ranking or create a grounded chart here."
        )
        findings = [
            f"Resolved metric: `{metric}`.",
            f"Resolved dimension: `{dimension}`.",
        ]
        next_steps = [
            "Run the query with access to the uploaded rows.",
            f"Then compute grouped `{metric}` by `{dimension}` and attach the resulting table/chart artifact.",
        ]
    else:
        return _continuity_context_response(question, conversation_context, data_source_context)

    limitations = [
        "Exact rankings need the raw uploaded rows, not only the saved profile.",
        "Small samples and extreme values can make a group look stronger than it really is.",
    ]
    timeline = _timeline("profile_based_analytical_answer", metric=metric, dimension=dimension, timestamp=timestamp)
    return _output(
        question=question,
        summary=summary,
        findings=findings,
        evidence=[f"Dataset profile for `{name}`.", "Detected column roles from the uploaded dataset profile."],
        limitations=limitations,
        next_steps=next_steps,
        code="",
        result_preview="",
        timeline=timeline,
        artifacts=[],
        trace_metadata={"fallback": "profile_based_analytical_answer", "data_source_name": name},
    )


def _continuity_context_response(
    question: str,
    conversation_context: Any,
    data_source_context: dict[str, Any],
) -> dict[str, Any]:
    context = conversation_context if isinstance(conversation_context, dict) else {}
    latest_findings = [
        str(item).strip()
        for item in context.get("latest_findings", [])
        if str(item).strip()
    ] if isinstance(context.get("latest_findings"), list) else []
    artifact_titles = [
        str(item).strip()
        for item in context.get("artifact_titles", [])
        if str(item).strip()
    ] if isinstance(context.get("artifact_titles"), list) else []
    name = str(data_source_context.get("name") or "selected data source")
    if latest_findings:
        strongest = latest_findings[0]
        chart_reference = f" Existing visual evidence includes {', '.join(artifact_titles[:2])}." if artifact_titles else ""
        summary = f"{strongest}{chart_reference} The useful validation is to test this claim with a direct metric-by-segment comparison, anomaly check, or chart tied to the same metric."
        findings = [
            "Candidate insight: " + item
            for item in latest_findings[:3]
        ]
    else:
        summary = (
            "There is not enough saved evidence to answer this follow-up with a concrete conclusion. "
            "The next analytical question should name a metric, segment, trend, anomaly, or chart target."
        )
        findings = ["No strong saved insights are available yet for this question."]

    if artifact_titles:
        findings.append("Useful existing outputs: " + ", ".join(artifact_titles[:4]) + ".")

    timeline = _timeline("investigation_continuity_context", findings=len(latest_findings), artifacts=len(artifact_titles))
    return _output(
        question=question,
        summary=summary,
        findings=findings,
        evidence=[
            "Current saved insights and analytical outputs.",
            f"Dataset context for `{name}`.",
        ],
        limitations=[
            "A fresh row-level calculation was not available for this answer.",
            "A precise chart or numeric comparison needs the uploaded CSV available to the analysis runtime.",
        ],
        next_steps=[
            "Run a concrete metric-by-segment comparison, trend, anomaly check, or chart.",
            "Use the strongest saved outputs as evidence if they support the conclusion.",
        ],
        code="",
        result_preview="",
        timeline=timeline,
        artifacts=[],
        trace_metadata={"fallback": "investigation_continuity_context", "data_source_name": name},
    )


def _findings_evidence_context_response(
    question: str,
    conversation_context: Any,
    data_source_context: dict[str, Any],
) -> dict[str, Any]:
    context = conversation_context if isinstance(conversation_context, dict) else {}
    state = context.get("conversation_state") if isinstance(context.get("conversation_state"), dict) else {}
    active_response = _active_target_evidence_output(question, state, context)
    if active_response:
        return active_response
    raw_findings = context.get("latest_findings")
    latest_findings = [str(item).strip() for item in raw_findings if str(item).strip()] if isinstance(raw_findings, list) else []
    report_summary = str(context.get("latest_report_summary") or "").strip()
    artifact_titles = [str(item).strip() for item in context.get("artifact_titles", []) if str(item).strip()] if isinstance(context.get("artifact_titles"), list) else []
    name = str(data_source_context.get("name") or "selected data source")

    if not latest_findings:
        summary = (
            "There are no substantive insights to audit yet. Start with a focused analysis, then review which conclusions "
            "need a supporting table, chart, or source reference."
        )
        findings = ["No current findings are available in this investigation."]
        evidence = [f"`{name}` does not yet have reviewed insight summaries."]
        next_steps = [
            "Ask a focused analytical question.",
            "After insights appear, attach the relevant table, chart, dataset, or report section as evidence.",
        ]
        result_rows: list[dict[str, str]] = []
    else:
        reviewed = latest_findings[:5]
        summary = (
            "The insights that most need evidence are the conclusions that are not yet backed by a table, chart, "
            f"or source reference. Prioritize {len(reviewed)} current insight{'' if len(reviewed) == 1 else 's'} for evidence review."
        )
        findings = [
            f"Needs evidence review: {item}"
            for item in reviewed
        ]
        evidence = [
            "Current investigation finding summaries.",
            "Linked artifact titles: " + ", ".join(artifact_titles[:5]) if artifact_titles else "No linked artifact titles were available in context.",
        ]
        if report_summary:
            evidence.append("Latest report summary was available for cross-checking.")
        next_steps = [
            "Attach the most relevant table, chart, dataset, or report section to each weak insight.",
            "Promote uncertain findings to an open question or risk if evidence is still weak.",
            "Resolve evidence gaps before finalizing the report.",
        ]
        result_rows = [{"finding": item, "recommended_action": "review and link supporting evidence"} for item in reviewed]

    limitations = [
        "This answer reviews saved insight summaries, not a new row-level scan of the raw CSV.",
        "Evidence completeness is not scored automatically yet; use linked evidence and reviewer judgment.",
    ]
    result_table = pd.DataFrame(result_rows)
    result_preview = result_table.to_string(index=False) if not result_table.empty else ""
    timeline = _timeline("data_source_context_findings_review", findings=len(latest_findings), artifacts=len(artifact_titles))
    return _output(
        question=question,
        summary=summary,
        findings=findings,
        evidence=evidence,
        limitations=limitations,
        next_steps=next_steps,
        code="",
        result_preview=result_preview,
        timeline=timeline,
        artifacts=[
            {
                "artifact_type": "table",
                "title": "Insight evidence review",
                "content": result_rows,
                "visibility": "user",
                "pinned": True,
                "metadata": {"source": "conversation_context"},
            }
        ] if result_rows else [],
        trace_metadata={"fallback": "data_source_context_findings_review", "data_source_name": name},
    )


def _active_target_evidence_output(
    question: str,
    state: dict[str, Any],
    conversation_context: dict[str, Any],
) -> dict[str, Any] | None:
    if not is_evidence_followup(question):
        return None
    active_target = state.get("active_analytical_target") or state.get("active_target")
    if not active_target:
        return None
    resolution = resolve_evidence_subject(
        question,
        active_target,
        branch_state=state,
        recent_findings=conversation_context.get("latest_findings") if isinstance(conversation_context.get("latest_findings"), list) else [],
    )
    if not resolution.target or resolution.score < 0.5:
        return None
    rows = _active_target_artifact_rows(resolution.target.to_payload(), conversation_context.get("recent_artifacts") or [])
    summary = evidence_response_text(question, resolution.target, artifact_rows=rows)
    return _output(
        question=question,
        summary=summary,
        findings=[summary],
        evidence=[
            "Resolved evidence against the latest active analytical target.",
            f"Resolution reason: {resolution.reason}; score={resolution.score:.2f}.",
        ],
        limitations=[
            "This follow-up is scoped to the active branch target; broader memory is only used when no active target exists.",
        ],
        next_steps=[],
        code="",
        result_preview=pd.DataFrame(rows[:12]).to_string(index=False) if rows else "",
        timeline=[{"tool": "evidence_resolution", "status": "ok", "reason": resolution.reason, "score": resolution.score}],
        artifacts=[],
        trace_metadata={
            "fallback": "evidence_resolution",
            "analysis_type": "evidence_resolution",
            "active_branch_type": resolution.target.branch_type,
            "metric": resolution.target.metric,
            "dimension": resolution.target.dimension,
            "hypothesis_mechanism": resolution.target.active_mechanism,
        },
    )


def _affected_findings_output(
    question: str,
    df: pd.DataFrame,
    state: dict[str, Any],
    conversation_context: dict[str, Any],
) -> dict[str, Any] | None:
    if not is_affected_findings_question(question):
        return None
    result = AffectedFindingsAnalyzer.analyze(
        question=question,
        dataframe=df,
        active_target=state.get("active_analytical_target") or state.get("active_target"),
        branch_state=state,
        recent_findings=[
            str(item)
            for item in (conversation_context.get("latest_findings") or [])
            if str(item).strip()
        ] if isinstance(conversation_context.get("latest_findings"), list) else [],
    )
    if not result:
        return None
    preview = pd.DataFrame(result.rows).to_string(index=False) if result.rows else ""
    return _output(
        question=question,
        summary=result.summary,
        findings=result.findings,
        evidence=result.evidence,
        limitations=result.limitations,
        next_steps=result.next_steps,
        code="",
        result_preview=preview,
        timeline=[{"tool": "affected_findings_analyzer", "status": "ok", "rows": len(result.rows)}],
        artifacts=[
            {
                "artifact_type": "table",
                "title": "Affected findings by quality issue",
                "content": result.rows,
                "visibility": "user",
                "pinned": True,
                "metadata": {
                    "analysis_type": "affected_findings",
                    "branch_type": "data_quality",
                    "evidence_role": "affected_findings",
                    "supports_current_target": True,
                },
            }
        ] if result.rows else [],
        trace_metadata={
            "fallback": "affected_findings_analyzer",
            "analysis_type": "affected_findings",
            "active_branch_type": "data_quality",
        },
    )


def _quality_branch_next_steps_output(question: str, state: dict[str, Any]) -> dict[str, Any] | None:
    if not _is_next_check_question(question):
        return None
    active_target = state.get("active_analytical_target") or state.get("active_target")
    resolution = resolve_evidence_subject(question, active_target, branch_state=state)
    target = resolution.target
    branch = str((target.branch_type if target else "") or state.get("active_branch_type") or "")
    mechanism = str((target.active_mechanism if target else "") or state.get("hypothesis_mechanism") or "")
    if branch != "data_quality" and mechanism != "duplicate_inflation":
        return None
    if target is None:
        target = build_active_target(question=question, state=state, trace_metadata={"active_branch_type": "data_quality"})
    summary = target_quality_next_steps(question, target)
    return _output(
        question=question,
        summary=summary,
        findings=[summary],
        evidence=["Continued the latest data-quality target instead of starting a generic grouped comparison."],
        limitations=["Exact duplicate source attribution may need load metadata or raw source row IDs."],
        next_steps=[],
        code="",
        result_preview="",
        timeline=[{"tool": "quality_branch_continuity", "status": "ok"}],
        artifacts=[],
        trace_metadata={
            "fallback": "quality_branch_continuity",
            "analysis_type": "data_quality",
            "active_branch_type": "data_quality",
            "metric": target.metric,
            "dimension": target.dimension,
        },
    )


def _active_target_artifact_rows(active_target: dict[str, Any], artifacts: Any) -> list[dict[str, Any]]:
    if not isinstance(artifacts, list):
        return []
    target_metric = str(active_target.get("metric") or "")
    target_dimension = str(active_target.get("dimension") or "")
    target_mechanism = str(active_target.get("active_mechanism") or "")
    for artifact in reversed(artifacts):
        if isinstance(artifact, dict):
            raw_content = artifact.get("content")
            raw_metadata = artifact.get("metadata")
        else:
            raw_content = getattr(artifact, "content", None)
            raw_metadata = getattr(artifact, "metadata", None)
        content = raw_content if isinstance(raw_content, dict) else {}
        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
        nested_metadata = metadata.get("metadata") if isinstance(metadata.get("metadata"), dict) else {}
        impact = content.get("transformation_impact") if isinstance(content.get("transformation_impact"), dict) else metadata.get("transformation_impact")
        if isinstance(impact, dict):
            impact_metric = str(impact.get("metric") or "")
            impact_dimension = str(impact.get("dimension") or "")
            if (not target_metric or not impact_metric or impact_metric == target_metric) and (
                not target_dimension or not impact_dimension or impact_dimension == target_dimension
            ):
                comparison_rows = impact.get("comparison_rows")
                if isinstance(comparison_rows, list) and comparison_rows:
                    return [row for row in comparison_rows if isinstance(row, dict)][:25]
        rows = content.get("rows") if isinstance(content.get("rows"), list) else raw_content
        if not isinstance(rows, list) or not rows:
            continue
        metric = str(metadata.get("metric") or nested_metadata.get("metric") or content.get("metric") or "")
        dimension = str(metadata.get("dimension") or nested_metadata.get("dimension") or content.get("dimension") or content.get("x") or "")
        mechanism = str(metadata.get("mechanism") or nested_metadata.get("mechanism") or metadata.get("operator") or nested_metadata.get("operator") or "")
        if target_metric and not metric:
            continue
        if target_dimension and not dimension:
            continue
        if target_metric and metric and metric != target_metric:
            continue
        if target_dimension and dimension and dimension != target_dimension:
            continue
        if target_mechanism and mechanism and mechanism != target_mechanism:
            continue
        if all(isinstance(row, dict) for row in rows):
            return rows
    return []




def _resolve_authoritative_focus(
    *,
    df: pd.DataFrame,
    question: str,
    conversation_context: dict[str, Any],
    semantic_profile: Any,
) -> tuple[str | None, str | None, str]:
    """Resolve metric/dimension with active investigation state as source of truth.

    Authority order:
    1. Explicit user instruction.
    2. Active investigation state.
    3. Latest chart/finding continuity.
    4. Semantic inference/defaults handled by the caller.
    """

    explicit_metric = _explicit_metric_column(df, question)
    explicit_dimension = _explicit_dimension_column(df, question, exclude={explicit_metric} if explicit_metric else set())
    value_match = match_entity_or_value(question, semantic_profile, df=df)
    if value_match:
        explicit_dimension = explicit_dimension or value_match.matched_column
    state = conversation_context.get("conversation_state") if isinstance(conversation_context.get("conversation_state"), dict) else {}
    chart = conversation_context.get("latest_chart_context") if isinstance(conversation_context.get("latest_chart_context"), dict) else {}
    intent = conversation_context.get("resolved_intent") if isinstance(conversation_context.get("resolved_intent"), dict) else {}
    primary_intent = str(intent.get("primary") or "").strip()
    active_metric = _valid_column(df, state.get("active_metric")) or _valid_column(df, chart.get("metric"))
    active_dimension = _valid_column(df, state.get("active_dimension")) or _valid_column(df, chart.get("dimension"))
    has_active_topic = bool(active_metric or active_dimension)
    explicit_topic_switch = bool(
        (explicit_metric and active_metric and explicit_metric != active_metric)
        or (explicit_dimension and active_dimension and explicit_dimension != active_dimension)
    )
    reset_requested = _is_investigation_reset_question(question) or _is_context_reset_question(question) or _is_business_questions_request(question) or _is_important_fields_request(question)

    high_priority_new_branch = _is_quality_question(question) or _is_hypothesis_validation_question(question)

    if reset_requested or high_priority_new_branch:
        return explicit_metric, explicit_dimension, "explicit_reset"
    if explicit_topic_switch:
        return explicit_metric or active_metric, explicit_dimension or active_dimension, "explicit_topic_switch"
    if explicit_metric or explicit_dimension:
        return explicit_metric or active_metric, explicit_dimension or active_dimension, "explicit_user_instruction"
    if has_active_topic and _should_bind_to_active_state(question, primary_intent):
        return active_metric, active_dimension, "active_investigation_state"
    return None, None, "semantic_inference"


def _should_bind_to_active_state(question: str, primary_intent: str) -> bool:
    if _is_business_questions_request(question) or _is_important_fields_request(question) or _is_context_reset_question(question):
        return False
    if is_transformation_question(question):
        return True
    if primary_intent in AUTHORITATIVE_FOLLOW_UP_TYPES:
        return True
    return _is_contextual_follow_up(question) or _is_volume_relationship_follow_up(question)




























































def _metric_coverage_response(question: str, df: pd.DataFrame) -> dict[str, Any] | None:
    available_columns = [str(col) for col in df.columns]
    timeline = _timeline("data_coverage_check", status="metric_missing", rows=len(df))
    summary = (
        "I do not see a usable numeric metric for this comparison. "
        "The available columns are better suited to profiling, grouping, or text/category review until a numeric field is selected."
    )
    return _output(
        question=question,
        summary=summary,
        findings=["No usable numeric metric column was found in the current dataset."],
        evidence=[f"Available columns: {', '.join(available_columns)}"],
        limitations=["A metric-oriented analysis requires at least one numeric column that is not identifier-like."],
        next_steps=[
            "Upload or select a dataset with at least one numeric metric column.",
            "Add semantic notes if a numeric-looking identifier should not be treated as a metric.",
        ],
        code="",
        result_preview="",
        timeline=timeline,
        artifacts=[
            {
                "artifact_type": "validation",
                "title": "Data coverage check",
                "content": timeline,
                "visibility": "technical",
            }
        ],
        trace_metadata={"fallback": "data_coverage_check", "available_columns": available_columns},
    )
