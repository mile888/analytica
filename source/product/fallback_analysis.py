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
    QueryFilter,
    TemporalSanityValidator,
)
from source.product.execution_context import (
    ExecutionContextUnavailableError,
    execution_context_failure_output,
    execution_required_for_question,
    execution_unavailable_in_context,
)
from source.product.direct_query_executor import DirectQueryExecutor
from source.product.metric_validation import validate_metric_for_analysis, validate_grouping
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
from source.product.question_routing import (
    GLOBAL_INTENTS,
    MULTI_DATASET_INTENTS,
    NON_ANALYTICAL_INTENTS,
    classify_question_intent,
    decide_routing,
    title_for_intent,
)
from source.product.fallbacks.semantic_resolution import (
    DELIVERY_DATE_MARKERS,
    ORDER_DATE_MARKERS,
    SHIPPING_METHOD_MARKERS,
    _contains_any,
    _dimension_explicitly_requested,
    _dimension_semantic_groups,
    _explicit_dimension_column,
    _explicit_metric_column,
    _find_duration_column,
    _has_correlation_candidate,
    _is_count_based_question,
    _looks_identifier_like,
    _looks_year_like_column,
    _mentioned_columns,
    _missing_explicit_entity,
    _missing_explicit_metric_request,
    _normalize,
    _parse_duration_series,
    _select_dimension_column,
    _select_metric_column,
    _select_timestamp_column,
    _semantic_column_by_markers,
    _semantic_column_match,
    _valid_column,
    _is_meaningful_dimension,
    resolve_categorical_value,
    detect_multi_label_column,
    explode_multi_label,
    resolve_semantic_alias,
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
    _is_duration_question,
    _is_duplicate_impact_followup,
    _is_duplicate_impact_question,
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
        findings=findings[:3],
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

    # --- MULTI-DATASET PREREQUISITE CHECK ---
    # Multi-dataset intents (joinability, warehouse, compare datasets, etc.) require
    # at least 2 loaded datasets. Since this function always operates on a single df,
    # check the data_context for the loaded dataset count. If < 2, refuse immediately.
    _prereq_intent = classify_question_intent(question)
    if _prereq_intent in MULTI_DATASET_INTENTS:
        _loaded_count = len((data_context or {}).get("data_source_ids") or []) if isinstance(data_context, dict) else 0
        # Also count attached_data_source_ids as a fallback
        if _loaded_count < 2:
            _loaded_count = max(_loaded_count, len((data_context or {}).get("attached_data_source_ids") or []) if isinstance(data_context, dict) else 0)
        # If we still have fewer than 2, the single df we have is the only dataset.
        if _loaded_count < 2:
            _loaded_count = max(_loaded_count, 1)  # We have at least the df
        if _loaded_count < 2:
            return _insufficient_dataset_scope_response(question, df, _prereq_intent, _loaded_count)

    conversation_context = (data_context or {}).get("conversation_context") if isinstance(data_context, dict) else {}
    df = _dataframe_with_derived_columns(df, conversation_context if isinstance(conversation_context, dict) else {})

    # ── Semantic Compatibility Guard (MUST run before ALL fallback paths) ──
    # This is the central enforcement point. No fallback path may answer
    # a question that is semantically incompatible with the dataset.
    from source.product.compatibility_engine import enforce_question_dataset_compatibility, build_incompatibility_output, check_question_dataset_compatibility
    _compat_decision = enforce_question_dataset_compatibility(question, df, context=data_context)
    if not _compat_decision.allowed:
        _compat_result = check_question_dataset_compatibility(question, df)
        if _compat_result and not _compat_result.compatible:
            return build_incompatibility_output(question, _compat_result)

    if _is_context_reset_question(question):
        return _dataframe_exploration_response(question, df)
    if _is_business_questions_request(question):
        return _business_questions_response(question, df)
    if _is_important_fields_request(question):
        return _important_fields_response(question, df)
    from source.product.business_semantic_planner import business_analysis_response
    business_response = business_analysis_response(
        question,
        df,
        conversation_context=conversation_context if isinstance(conversation_context, dict) else {},
    )
    if business_response:
        return business_response
    if _is_strategic_synthesis_question(question):
        return _strategic_synthesis_response(question, df)
    if _is_overview_question(question) and _is_dependency_question(question):
        return _dataframe_overview_with_dependencies_response(question, df)
    if _is_overview_question(question):
        return _dataframe_exploration_response(question, df)
    if _is_ingestion_diagnostic_question(question):
        return _ingestion_diagnostic_response(question, df)

    early_state = conversation_context.get("conversation_state") if isinstance(conversation_context, dict) and isinstance(conversation_context.get("conversation_state"), dict) else {}
    # --- Refusal-continuation guard ---
    # If the previous analysis was a refusal and the current question is a continuation-style
    # follow-up, do NOT fall back into stale grouped analysis.
    _REFUSAL_TYPES = {"hard_stop", "hard_stop_missing_fields", "missing_explicit_metric", "insufficient_dataset_scope", "fallback"}
    _prior_analysis_type = str(early_state.get("analysis_type") or early_state.get("last_error_type") or "")
    if _prior_analysis_type in _REFUSAL_TYPES and _is_continuation_after_refusal(question):
        return _output(
            question=question,
            summary=(
                "There is no active analysis to continue because the previous request could not be executed. "
                "Please start a new question using available fields, or rephrase the original request."
            ),
            findings=[],
            evidence=["The previous analysis ended with a refusal due to missing or unavailable fields."],
            limitations=["Continuation-style follow-ups require a successful prior analysis to build on."],
            next_steps=["Ask a new analytical question using available fields in this dataset."],
            code="",
            result_preview="",
            timeline=_timeline("continuation_after_refusal"),
            artifacts=[],
            trace_metadata={
                "fallback": "continuation_after_refusal",
                "analysis_type": "continuation_after_refusal",
                "suppress_key_findings": True,
            },
        )
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
    plan_patch = _followup_plan_patch_response(question, df, conversation_context if isinstance(conversation_context, dict) else {})
    if plan_patch:
        return plan_patch
    explicit_semantic = _explicit_semantic_plan_response(question, df, conversation_context if isinstance(conversation_context, dict) else {})
    if explicit_semantic:
        return explicit_semantic
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
    # ── SEMANTIC ROLE ASSIGNMENT INTERCEPT ──
    # For prevalence, association, and outcome-based questions, infer the
    # correct target/outcome vs. grouping variables BEFORE any metric/dimension
    # selection. This prevents role confusion (e.g. mean(grouping_flag) vs
    # mean(outcome_variable) grouped by grouping_flag).
    from source.product.semantic_role_assignment import (
        assign_semantic_roles,
        prevalence_analysis_response,
        association_analysis_response,
    )
    if _is_factor_diversity_question(question):
        _factor_diversity = _factor_diversity_response(question, df)
        if _factor_diversity:
            return _factor_diversity
    if _is_ordered_indicator_gradient_question(question):
        _indicator_gradient = _ordered_indicator_gradient_response(question, df)
        if _indicator_gradient:
            return _indicator_gradient
    if _is_binary_indicator_prevalence_question(question):
        _indicator_prevalence = _binary_indicator_prevalence_response(question, df)
        if _indicator_prevalence:
            return _indicator_prevalence
    if _is_ordered_group_prevalence_question(question):
        _ordered_prevalence = _ordered_group_prevalence_response(question, df)
        if _ordered_prevalence:
            return _ordered_prevalence
    _roles = assign_semantic_roles(question, df)
    if _roles and _roles.confidence >= 0.5:
        if _roles.operation == "prevalence" and _roles.target_variable:
            _prevalence_result = prevalence_analysis_response(question, df, _roles)
            if _prevalence_result:
                return _prevalence_result
        elif _roles.operation == "association" and _roles.target_variable:
            _association_result = association_analysis_response(question, df, _roles)
            if _association_result:
                return _association_result
    # --- EARLY OPERATION ROUTING ---
    # Specialized analysis patterns that should be handled by dedicated handlers
    # instead of the authoritative planner (which may reject them for missing fields).
    if _is_duration_by_category_question(question):
        dur_col = _find_duration_column(df)
        cat_col_for_dur = _find_category_col_from_question(question, df)
        if dur_col and cat_col_for_dur:
            return _duration_by_category_response(question, df, dur_col, cat_col_for_dur)
    # --- MULTI-FACTOR ASSOCIATION INTERCEPT ---
    # "Which factors are most associated with target?" → ranked factor analysis
    if _is_multi_factor_question(question):
        return _multi_factor_association_response(question, df)
    # --- CATEGORY GROWTH INTERCEPT ---
    # "What categories are growing fastest?" → growth rate by period
    if _is_growth_question(question) and not _explicit_metric_column(df, question):
        year_col_for_growth = _find_year_col(df)
        cat_col_for_growth = _find_category_col_from_question(question, df)
        if year_col_for_growth and cat_col_for_growth:
            return _category_growth_response(question, df, cat_col_for_growth, year_col_for_growth)
    if _is_temporal_shift_question(question) and not _explicit_metric_column(df, question):
        year_col_for_shift = _find_year_col(df)
        cat_col_for_shift = _find_category_col_from_question(question, df)
        if year_col_for_shift and cat_col_for_shift:
            return _temporal_mix_shift_response(question, df, cat_col_for_shift, year_col_for_shift)
    if _is_strategic_synthesis_question(question) and not _is_temporal_shift_question(question) and not _explicit_metric_column(df, question):
        return _strategic_synthesis_response(question, df)
    authoritative = _authoritative_plan_output(question, df, conversation_context if isinstance(conversation_context, dict) else {})
    if authoritative:
        return authoritative
    # --- GUARD 2: Global / Multi-Dataset Intent Hard Bypass ---
    _guard_intent = classify_question_intent(question)
    if _guard_intent in GLOBAL_INTENTS or _guard_intent in NON_ANALYTICAL_INTENTS:
        _guard_routing = decide_routing(question, has_active_context=bool((conversation_context or {}).get("conversation_state")))
        return _global_intent_response(question, df, _guard_routing)
    missing_metric = _missing_explicit_metric_request(question, [str(col) for col in df.columns])
    if missing_metric and not (is_hypothesis_question(question) or _is_hypothesis_validation_question(question)):
        semantic = build_semantic_dataset_profile(df=df)
        ranked_metrics = [item["name"] for item in _rank_metric_candidates(df, semantic)[:6]]
        _registry_entries = (data_context or {}).get("dataset_registry") if isinstance(data_context, dict) else None
        return _missing_metric_response(question, missing_metric, ranked_metrics, source_name="the dataset", dataset_registry=_registry_entries)
    # --- GUARD 1: Missing Explicit Requirements Hard Stop ---
    # Runs after the language-aware _missing_metric_response (which handles most missing-metric cases).
    # This guard catches remaining cases where the planner detects missing metric/dimension.
    _guard_plan = AuthoritativeExecutionPlanner.plan(question, df)
    _explicit_missing = [f for f in _guard_plan.missing_required_fields if f in {"metric", "dimension"}]
    if _explicit_missing and not _is_count_based_question(question):
        return _hard_stop_missing_fields(question, df, _guard_plan)
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
    # --- EARLY COUNT-BASED INTERCEPT ---
    # When the question implies counting (most common, frequency, dominance) and no metric was
    # explicitly requested by the user, prefer count-based analysis. This prevents frequency
    # questions from being routed to grouped metric analysis using arbitrary numeric columns.
    _explicit_metric_in_question = _explicit_metric_column(df, question) is not None
    # --- TEMPORAL CATEGORY MIX SHIFT INTERCEPT ---
    # "How did content strategy change after 2018?" → before/after category distribution
    # Must fire BEFORE count-based because temporal questions often overlap with count patterns.
    if _is_temporal_shift_question(question) and not _explicit_metric_in_question:
        year_col_for_shift = _find_year_col(df)
        cat_col_for_shift = _find_category_col_from_question(question, df)
        if year_col_for_shift and cat_col_for_shift:
            return _temporal_mix_shift_response(question, df, cat_col_for_shift, year_col_for_shift)
    # --- DIVERSITY ANALYSIS INTERCEPT ---
    # "Compare regional content diversity" → unique categories by group
    if _is_diversity_question(question):
        entity_col = _find_category_col_from_question(question, df)
        group_col = _find_group_col_from_question(question, df, exclude=entity_col)
        if entity_col and group_col:
            return _diversity_analysis_response(question, df, group_col, entity_col)
    # --- DURATION-BY-CATEGORY INTERCEPT ---
    # "Compare average durations by genre" → parsed duration by category
    if _is_duration_by_category_question(question):
        dur_col = _find_duration_column(df)
        cat_col_for_dur = _find_category_col_from_question(question, df)
        if dur_col and cat_col_for_dur:
            return _duration_by_category_response(question, df, dur_col, cat_col_for_dur)
    # --- STRATEGIC SYNTHESIS INTERCEPT ---
    # "Summarize the platform strategy" / "strategically important" → computed evidence synthesis
    if _is_strategic_synthesis_question(question) and not _explicit_metric_in_question:
        return _strategic_synthesis_response(question, df)
    # --- EARLY COUNT-BASED INTERCEPT ---
    # When the question implies counting (most common, frequency, dominance) and no metric was
    # explicitly requested by the user, prefer count-based analysis.
    if _is_count_based_question(question) and not _explicit_metric_in_question and focus_authority in {"semantic_inference", "explicit_user_instruction"}:
        count_dimension = _count_based_dimension(question, df, dimension_col, timestamp_col)
        if count_dimension:
            return _count_based_response(question, df, count_dimension)
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
        # --- COUNT-BASED ANALYSIS ---
        # If the question implies counting (releases by year, distribution, dominance), use record count.
        if _is_count_based_question(question):
            # Check if the question mentions a year or timestamp column as the grouping axis.
            count_dimension = _count_based_dimension(question, df, dimension_col, timestamp_col)
            if count_dimension:
                return _count_based_response(question, df, count_dimension)
        if dimension_col and _is_chart_question(question):
            count_dimension = _count_based_dimension(question, df, dimension_col, timestamp_col)
            if count_dimension:
                return _count_based_response(question, df, count_dimension)
        # --- DURATION PARSING ---
        # If question mentions duration/length and a text column has duration values, parse it.
        if _is_duration_question(question):
            duration_col = _find_duration_column(df)
            if duration_col:
                parsed = _parse_duration_series(df[duration_col])
                if parsed is not None:
                    working = df.copy()
                    working["_parsed_duration"] = parsed
                    working = working.dropna(subset=["_parsed_duration"])
                    if not working.empty:
                        if _is_outlier_question(question):
                            return _outlier_response(question, working, "_parsed_duration", original_name=duration_col)
                        metric_col = "_parsed_duration"
                        # Continue with normal flow using parsed duration
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
        return _ensure_chart_artifact(_single_metric_response(question, df, metric_col), question)

    # --- GUARD 3: Downstream Grouped Fallback Safety Net ---
    _final_intent = classify_question_intent(question)
    if _final_intent in GLOBAL_INTENTS or _final_intent in NON_ANALYTICAL_INTENTS:
        return _global_intent_response(question, df, decide_routing(question))
    return _ensure_chart_artifact(_grouped_metric_response(question, df, metric_col, dimension_col), question)


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


def _is_continuation_after_refusal(question: str) -> bool:
    """Detect continuation-style follow-up questions that reference a prior analysis."""
    text = _normalize(question)
    continuation_markers = (
        "compare against",
        "compare with",
        "compare to",
        "and what about",
        "what about the",
        "now filter by",
        "now show",
        "now compare",
        "continue with",
        "follow up",
        "сравни с",
        "а если сравнить",
        "а что с",
        "теперь покажи",
        "теперь сравни",
        "продолжи",
    )
    return any(marker in text for marker in continuation_markers)


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
        findings=findings[:3],
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
    if _is_true_dtype_inspection_question(text) and not _has_explicit_analytical_operation_intent(text):
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


def _explicit_semantic_plan_response(question: str, df: pd.DataFrame, conversation_context: dict[str, Any]) -> dict[str, Any] | None:
    try:
        from source.product.llm_semantic_planner import plan_explicit_constraints
        from source.product.plan_validator import PlanRejection, validate_plan
        from source.product.plan_executor import execute_plan
        from source.product.grounded_synthesis import mechanical_synthesis
    except Exception:
        return None

    plan = plan_explicit_constraints(question, df, context=conversation_context)
    if not plan:
        return None
    validated = validate_plan(plan, df)
    if isinstance(validated, PlanRejection):
        return _output(
            question=question,
            summary="I preserved the explicit analytical constraints, but the plan cannot be executed: " + "; ".join(validated.reasons),
            findings=[],
            evidence=["Explicit constraint validation rejected the plan before execution."],
            limitations=validated.reasons,
            next_steps=["Use fields that exist in this dataset and are compatible with the requested aggregation."],
            code="",
            result_preview="",
            timeline=[{"tool": "explicit_constraint_semantic_planner", "status": "rejected", "reasons": validated.reasons}],
            artifacts=[],
            trace_metadata={
                "fallback": "explicit_constraint_semantic_planner",
                "analysis_type": "plan_validation_rejection",
                "semantic_planner": True,
                "query_plan": plan.to_dict(),
                "suppress_key_findings": True,
            },
        )

    evidence = execute_plan(validated, df, question=question)
    if not evidence.computed_tables and not evidence.summary_statistics and not evidence.artifacts:
        return None
    synthesis = mechanical_synthesis(question, evidence)
    artifacts = []
    for artifact in evidence.artifacts:
        if not isinstance(artifact, dict):
            continue
        prepared = dict(artifact)
        prepared.setdefault("artifact_type", prepared.get("type") or "table")
        metadata = prepared.get("metadata") if isinstance(prepared.get("metadata"), dict) else {}
        metadata.setdefault("analysis_type", plan.operation.lower())
        metadata.setdefault("semantic_planner", True)
        metadata.setdefault("query_plan", plan.to_dict())
        prepared["metadata"] = metadata
        artifacts.append(prepared)
    preview_rows: list[dict[str, Any]] = []
    if evidence.computed_tables:
        rows = evidence.computed_tables[0].get("rows")
        preview_rows = rows if isinstance(rows, list) else []
    result_preview = pd.DataFrame(preview_rows).head(20).to_string(index=False) if preview_rows else ""
    findings = []
    for finding in synthesis.findings:
        if isinstance(finding, dict):
            title = str(finding.get("title") or "").strip()
            evidence_text = str(finding.get("evidence") or "").strip()
            findings.append(f"{title}: {evidence_text}" if title and evidence_text else title or evidence_text)
        else:
            findings.append(str(finding))
    plan_payload = plan.to_dict()
    if plan.operation == "GROUPED_AGGREGATION":
        plan_payload.setdefault("intent", "rank_groups")
    return _output(
        question=question,
        summary=synthesis.answer,
        findings=[item for item in findings if item],
        evidence=[
            f"Validated explicit semantic plan: operation={plan.operation}, metric={plan.metric}, groups={plan.grouping_columns}, aggregation={plan.aggregation}.",
            *evidence.validation_notes,
        ],
        limitations=[*evidence.limitations, *synthesis.limitations],
        next_steps=synthesis.next_steps,
        code="deterministic semantic plan execution",
        result_preview=result_preview,
        timeline=[
            {"tool": "explicit_constraint_semantic_planner", "status": "ok", "operation": plan.operation, "constraints_locked": plan.constraints_locked},
            {"tool": "plan_validator", "status": "ok", "repairs": validated.repairs},
            {"tool": "deterministic_plan_executor", "status": "ok", "record_count": evidence.record_count},
            {"tool": "grounded_mechanical_synthesis", "status": "ok"},
        ],
        artifacts=artifacts,
        trace_metadata={
            "fallback": "explicit_constraint_semantic_planner",
            "analysis_type": "rank_groups" if plan.operation == "GROUPED_AGGREGATION" else plan.operation.lower(),
            "semantic_planner": True,
            "planner_operation": plan.operation,
            "constraints_locked": plan.constraints_locked,
            "query_plan": plan_payload,
            "columns_used": evidence.columns_used,
            "artifact_required": plan.artifact_required,
        },
    )


def _followup_plan_patch_response(question: str, df: pd.DataFrame, conversation_context: dict[str, Any]) -> dict[str, Any] | None:
    if not _is_followup_plan_patch_question(question):
        return None
    base = _previous_grouped_plan_context(conversation_context)
    if not base:
        return None
    metric = _explicit_metric_column(df, question) or str(base.get("metric") or "").strip()
    dimension = _explicit_dimension_column(df, question, exclude={metric} if metric else set()) or str(base.get("dimension") or "").strip()
    if not metric or not dimension or metric not in df.columns or dimension not in df.columns:
        return None

    aggregation = _aggregation_override(question)
    if not aggregation:
        aggregation = str(base.get("aggregation") or "sum").strip() or "sum"
    limit = _limit_override(question)
    if limit is None:
        limit = _int_or_none(base.get("limit") or base.get("top_n"))
    chart_type = _chart_type_override(question) or str(base.get("chart_type") or "bar").strip() or "bar"
    filters = _patched_filters(question, df, dimension, metric, base)
    plan = AuthoritativeQueryPlan(
        raw_question=question,
        intent=str(base.get("intent") or "rank_groups"),
        metric=metric,
        metric_source="followup_patch",
        metric_alias_used=None,
        dimension=dimension,
        dimension_source="followup_patch",
        filters=filters,
        aggregation=aggregation,
        aggregation_intent="followup_override" if _aggregation_override(question) else str(base.get("aggregation_intent") or "previous_context"),
        target_entity=dimension,
        ranking_direction=str(base.get("ranking_direction") or base.get("ranking_mode") or "descending"),
        limit=limit,
        chart_type=chart_type,
        scope="follow_up",
        requires_new_branch=True,
        can_use_active_branch=False,
        confidence="high",
    )
    workspace = BranchWorkspaceManager.from_payload(
        ((conversation_context.get("conversation_state") or {}).get("branch_workspace") if isinstance(conversation_context, dict) else None)
    )
    route = BranchWorkspaceManager.route(plan, workspace)
    result = _execute_rank_groups_plan(question, df, plan, route)
    if not result:
        return None
    trace = result.setdefault("trace_metadata", {})
    if isinstance(trace, dict):
        trace["fallback"] = "followup_plan_patch"
        trace["operation"] = "FOLLOWUP_PLAN_PATCH"
        trace["patched_from"] = base.get("source") or "active_chart_context"
    for artifact in result.get("artifacts", []) if isinstance(result.get("artifacts"), list) else []:
        if not isinstance(artifact, dict):
            continue
        if aggregation == "sum" and artifact.get("artifact_type") in {"table", "chart"}:
            artifact["title"] = f"Total {metric} by {dimension}"
        elif aggregation == "mean" and artifact.get("artifact_type") in {"table", "chart"}:
            artifact["title"] = f"Average {metric} by {dimension}"
        elif aggregation == "median" and artifact.get("artifact_type") in {"table", "chart"}:
            artifact["title"] = f"Median {metric} by {dimension}"
        elif aggregation == "count" and artifact.get("artifact_type") in {"table", "chart"}:
            artifact["title"] = f"Count of {metric} by {dimension}"
        metadata = artifact.get("metadata") if isinstance(artifact.get("metadata"), dict) else {}
        for key in ("dataset_id", "dataset_ids", "dataset_scope"):
            if base.get(key) and key not in metadata:
                metadata[key] = base[key]
        artifact["metadata"] = metadata
    return result


def _is_followup_plan_patch_question(question: str) -> bool:
    text = _normalize(question)
    correction_markers = (
        "do not use",
        "don't use",
        "dont use",
        "instead",
        "switch to",
        "change to",
        "same but",
        "not average",
        "not mean",
        "not total",
        "use total",
        "use average",
        "use mean",
        "use median",
        "use count",
        "use sum",
    )
    override_markers = ("average", "mean", "total", "sum", "median", "count", "top ", "bar", "line", "chart", "group", "by ")
    return any(marker in text for marker in correction_markers) and any(marker in text for marker in override_markers)


def _previous_grouped_plan_context(conversation_context: dict[str, Any]) -> dict[str, Any] | None:
    state = conversation_context.get("conversation_state") if isinstance(conversation_context.get("conversation_state"), dict) else {}
    artifacts = list(conversation_context.get("recent_artifacts") or [])
    for artifact in reversed(artifacts):
        context = _plan_context_from_artifact(artifact)
        if context:
            return context
    chart = conversation_context.get("latest_chart_context") if isinstance(conversation_context.get("latest_chart_context"), dict) else {}
    if chart:
        context = _plan_context_from_chart(chart)
        if context:
            return context
    if state:
        context = _plan_context_from_chart(state)
        if context:
            return context
    return None


def _plan_context_from_artifact(artifact: Any) -> dict[str, Any] | None:
    artifact_type = str(artifact.get("artifact_type") if isinstance(artifact, dict) else getattr(getattr(artifact, "artifact_type", None), "value", getattr(artifact, "artifact_type", "")) or "")
    if artifact_type != "chart":
        return None
    content = artifact.get("content") if isinstance(artifact, dict) else getattr(artifact, "content", None)
    metadata = artifact.get("metadata") if isinstance(artifact, dict) else getattr(artifact, "metadata", None)
    content = content if isinstance(content, dict) else {}
    metadata = metadata if isinstance(metadata, dict) else {}
    plan_payload = metadata.get("query_plan") if isinstance(metadata.get("query_plan"), dict) else content.get("query_plan")
    if isinstance(plan_payload, dict):
        context = dict(plan_payload)
    else:
        context = {}
    context.update(
        {
            "source": "recent_chart_artifact",
            "metric": context.get("metric") or metadata.get("metric") or content.get("metric") or content.get("y"),
            "dimension": context.get("dimension") or metadata.get("dimension") or content.get("dimension") or content.get("x"),
            "aggregation": context.get("aggregation") or metadata.get("aggregation") or content.get("aggregation") or _aggregation_from_axis(content.get("y")),
            "filters": context.get("filters") or metadata.get("filters") or content.get("filters") or [],
            "chart_type": context.get("chart_type") or metadata.get("chart_type") or content.get("chart_type"),
            "limit": context.get("limit") or metadata.get("top_n") or _rows_limit(content.get("rows")),
            "top_n": metadata.get("top_n") or context.get("limit") or _rows_limit(content.get("rows")),
            "dataset_id": metadata.get("dataset_id") or "",
            "dataset_ids": metadata.get("dataset_ids") or [],
            "dataset_scope": metadata.get("dataset_scope") or "",
        }
    )
    return context if context.get("metric") and context.get("dimension") else None


def _plan_context_from_chart(chart: dict[str, Any]) -> dict[str, Any] | None:
    metric = chart.get("metric") or chart.get("active_metric")
    dimension = chart.get("dimension") or chart.get("active_dimension")
    if not metric or not dimension:
        return None
    return {
        "source": "latest_chart_context",
        "intent": "rank_groups",
        "metric": metric,
        "dimension": dimension,
        "aggregation": chart.get("aggregation") or chart.get("active_aggregation") or chart.get("active_chart_aggregation") or "sum",
        "filters": chart.get("filters") or chart.get("active_filters") or [],
        "chart_type": chart.get("chart_type") or chart.get("active_chart_type") or "bar",
        "limit": chart.get("top_n") or chart.get("displayed_rows"),
        "top_n": chart.get("top_n") or chart.get("displayed_rows"),
        "ranking_direction": chart.get("ranking_direction") or chart.get("ranking_mode") or "descending",
        "dataset_id": chart.get("dataset_id") or chart.get("active_dataset_id") or "",
        "dataset_ids": chart.get("dataset_ids") or chart.get("active_dataset_ids") or [],
        "dataset_scope": chart.get("dataset_scope") or chart.get("active_dataset_scope") or "",
    }


def _aggregation_override(question: str) -> str | None:
    text = _normalize(question)
    matches: list[tuple[int, str]] = []
    marker_map = {
        "sum": ("total", "summed", "sum", "суммар", "сумма", "всего"),
        "mean": ("average", "mean", "avg", "средн"),
        "median": ("median", "медиан"),
        "count": ("count", "volume", "record count", "колич", "объем", "объём"),
    }
    for aggregation, markers in marker_map.items():
        for marker in markers:
            start = 0
            while True:
                idx = text.find(marker, start)
                if idx < 0:
                    break
                if not _marker_is_negated(text, idx):
                    matches.append((idx, aggregation))
                start = idx + len(marker)
    if not matches:
        return None
    matches.sort(key=lambda item: item[0])
    return matches[-1][1]


def _marker_is_negated(text: str, idx: int) -> bool:
    prefix = text[max(0, idx - 24):idx]
    return any(marker in prefix for marker in ("don't use", "dont use", "do not use", "not ", "no "))


def _limit_override(question: str) -> int | None:
    match = re.search(r"\btop\s+(\d{1,3})\b", _normalize(question))
    if not match:
        return None
    try:
        value = int(match.group(1))
    except ValueError:
        return None
    return max(1, min(value, 100))


def _chart_type_override(question: str) -> str | None:
    text = _normalize(question)
    if "heatmap" in text:
        return "heatmap"
    if "scatter" in text:
        return "scatter"
    if "line" in text:
        return "line"
    if "bar" in text or "chart" in text or "plot" in text or "graph" in text:
        return "bar"
    return None


def _patched_filters(question: str, df: pd.DataFrame, dimension: str, metric: str, base: dict[str, Any]) -> list[QueryFilter]:
    text = _normalize(question)
    if any(marker in text for marker in ("remove filter", "clear filter", "without filter", "all records", "all rows", "no filter")):
        return []
    base_filters = _query_filters_from_payload(base.get("filters"))
    if not any(marker in text for marker in (" filter ", " only ", " where ", " for ", " in ")):
        return base_filters
    preferred = [dimension] + [str(col) for col in df.columns if str(col) not in {dimension, metric}]
    metric_terms = {_normalize(metric), *_normalize(metric).split()}
    resolved = resolve_categorical_value(question, df, preferred_columns=preferred, exclude=metric_terms)
    if not resolved:
        return base_filters
    return [QueryFilter(column=resolved.column, operator="equals", value=resolved.value, source=resolved.source)]


def _query_filters_from_payload(payload: Any) -> list[QueryFilter]:
    filters: list[QueryFilter] = []
    for item in payload if isinstance(payload, list) else []:
        if not isinstance(item, dict):
            continue
        column = str(item.get("column") or "").strip()
        if not column:
            continue
        filters.append(QueryFilter(column=column, operator=str(item.get("operator") or "equals"), value=item.get("value"), source=str(item.get("source") or "previous_context")))
    return filters


def _aggregation_from_axis(axis: Any) -> str:
    value = str(axis or "").strip().lower()
    if value == "total":
        return "sum"
    if value == "avg":
        return "mean"
    return value if value in {"sum", "mean", "median", "count"} else ""


def _rows_limit(rows: Any) -> int | None:
    return len(rows) if isinstance(rows, list) and rows else None


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return max(1, min(int(value), 100))
    except (TypeError, ValueError):
        return None


def _is_true_dtype_inspection_question(normalized_question: str) -> bool:
    """Dtype inspection is only for schema/type/convertibility requests."""
    explicit_type_markers = (
        "dtype", "dtypes", "data type", "data types", "column type", "column types",
        "type of column", "types of columns", "inspect types", "inspect dtypes",
        "show data types", "show column types", "check dataframe schema",
        "dataframe schema", "schema", "convertibility", "converted to numeric",
        "can be converted to numeric", "convert to numeric", "convert numeric",
        " is numeric", " are numeric", "тип", "типы", "числов",
    )
    numeric_column_markers = (
        "which columns are numeric",
        "what columns are numeric",
        "numeric columns",
        "числовые колонки",
    )
    return any(marker in normalized_question for marker in explicit_type_markers) or any(
        marker in normalized_question for marker in numeric_column_markers
    )


def _has_explicit_analytical_operation_intent(normalized_question: str) -> bool:
    """Analytical ranking/aggregation/chart intent must win over schema wording."""
    analytical_markers = (
        "top ", "top-", "highest", "lowest", "rank", "ranking",
        " by total ", " total ", "total by", "sum ", "sum of",
        "average ", "average of", "mean ", "mean of", "median ",
        "group by", "grouping column", "grouped by", " by ",
        "metric", "numeric metric", "compare", "comparison",
        "distribution", "chart", "visualize", "visualise", "plot", "graph",
        "show top", "build a chart", "create a chart",
    )
    return any(marker in normalized_question for marker in analytical_markers)


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
    # --- GUARD: Off-topic question detection ---
    # If the user asks about a domain (films, food, etc.) that doesn't exist
    # in the dataset and the plan used pure defaults, warn instead of silently
    # substituting unrelated columns.
    off_topic = _detect_off_topic_question(question, df, plan)
    if off_topic:
        semantic = build_semantic_dataset_profile(df=df)
        numeric_cols = [item.name for item in semantic.metrics[:6]]
        categorical_cols = [item.name for item in semantic.dimensions[:6]]
        parts = [off_topic]
        if numeric_cols:
            parts.append(f"Available numeric fields: {', '.join(f'`{col}`' for col in numeric_cols)}.")
        if categorical_cols:
            parts.append(f"Available grouping fields: {', '.join(f'`{col}`' for col in categorical_cols)}.")
        parts.append("Rephrase the question using the fields available in this dataset, or upload a dataset that contains the requested data.")
        return _output(
            question=question,
            summary=" ".join(parts),
            findings=[],
            evidence=["The question references concepts not found in the current dataset schema."],
            limitations=["No matching fields for the requested subject."],
            next_steps=["Rephrase using available field names or upload a matching dataset."],
            code="",
            result_preview="",
            timeline=_timeline("hard_stop_off_topic", question=question[:100]),
            artifacts=[],
            trace_metadata={
                "fallback": "hard_stop_off_topic",
                "analysis_type": "hard_stop",
                "query_plan": plan.to_payload(),
                "suppress_key_findings": True,
            },
        )
    # --- GUARD: Refuse distant alias substitution ---
    # If the metric was resolved via safe_alias but the original requested name
    # does not appear in the resolved column name, this is a distant substitution
    # between unrelated metric names. Block execution.
    # We allow cross-language aliases within the same semantic group because
    # the column name contains a marker from that semantic alias group.
    if plan.metric_source == "safe_alias" and plan.metric_alias_used and plan.metric:
        alias_lower = _normalize(plan.metric_alias_used).strip("?!.,;:\"'()[]{}#")
        metric_lower = _normalize(plan.metric)
        columns_lower = [_normalize(str(col)) for col in df.columns]
        # Check 1: user's exact word in the resolved column name or vice versa
        alias_in_resolved = alias_lower in metric_lower or metric_lower in alias_lower
        # Check 2: user's exact word in ANY column name in the dataset
        alias_in_any_column = any(alias_lower in col for col in columns_lower)
        # Check 3: non-Latin script trusts the resolver (cross-language alias)
        has_non_latin = any(ord(ch) > 127 for ch in alias_lower)
        # Check 4: the resolved column name STARTS with a known alias marker
        # from the same group, while prefixed unrelated names remain blocked.
        column_is_clean_alias = False
        if not alias_in_resolved and not alias_in_any_column and not has_non_latin:
            from source.product.execution_planner import SafeBusinessAliasResolver
            for _group_markers in SafeBusinessAliasResolver.METRIC_ALIAS_GROUPS.values():
                if any(marker in alias_lower or alias_lower in marker for marker in _group_markers):
                    # The user's phrase matches this alias group.
                    # Check if the resolved column name starts with any marker from this group.
                    metric_parts = metric_lower.split()
                    first_word = metric_parts[0] if metric_parts else ""
                    column_is_clean_alias = any(
                        first_word.startswith(marker) or marker.startswith(first_word)
                        for marker in _group_markers
                    )
                    break
        is_close = alias_in_resolved or alias_in_any_column or has_non_latin or column_is_clean_alias
        if not is_close:
            # The resolved metric is semantically distant from what the user asked for.
            semantic = build_semantic_dataset_profile(df=df)
            numeric_cols = [item.name for item in semantic.metrics[:8]] if hasattr(semantic, "metrics") else [str(col) for col in df.select_dtypes(include="number").columns[:8]]
            categorical_cols = [item.name for item in semantic.dimensions[:8]] if hasattr(semantic, "dimensions") else [str(col) for col in df.select_dtypes(exclude="number").columns[:8]]
            parts = []
            _registry_entries = (conversation_context or {}).get("dataset_registry") if isinstance(conversation_context, dict) else None
            other_note = _other_dataset_has_field(plan.metric_alias_used, _registry_entries)
            if other_note:
                parts.append(other_note)
            else:
                parts.append(f"I cannot build this analysis because the current dataset does not contain a `{plan.metric_alias_used}` field.")
            parts.append(f"I found a field called `{plan.metric}`, but it does not appear to be a direct `{plan.metric_alias_used}` field, so I did not substitute it automatically.")
            if numeric_cols:
                parts.append(f"Available numeric fields: {', '.join(f'`{col}`' for col in numeric_cols)}.")
            if categorical_cols:
                parts.append(f"Available categorical fields: {', '.join(f'`{col}`' for col in categorical_cols)}.")
            return _output(
                question=question,
                summary=" ".join(parts),
                findings=[],
                evidence=[f"Alias `{plan.metric_alias_used}` resolved to `{plan.metric}` — too distant to substitute silently."],
                limitations=[f"No `{plan.metric_alias_used}` field was available; `{plan.metric}` is not a safe substitute."],
                next_steps=["Rephrase the question using available field names or upload a dataset with the requested fields."],
                code="",
                result_preview="",
                timeline=_timeline("hard_stop_distant_alias", alias=plan.metric_alias_used, resolved=plan.metric),
                artifacts=[],
                trace_metadata={
                    "fallback": "hard_stop_missing_fields",
                    "analysis_type": "hard_stop",
                    "query_plan": plan.to_payload(),
                    "suppress_key_findings": True,
                },
            )
    if plan.intent == "meta":
        return None
    if plan.hypothesis:
        return None
    if is_transformation_question(question) or _is_hypothesis_validation_question(question):
        return None
    metric_check = validate_metric_for_analysis(plan.metric, df)
    if not metric_check.valid and plan.metric:
        if metric_check.suggested_alternative:
            plan = replace(plan, metric=metric_check.suggested_alternative, metric_source="validation_override")
        else:
            semantic = build_semantic_dataset_profile(df=df)
            numeric_cols = [item.name for item in semantic.metrics[:8]]
            categorical_cols = [item.name for item in semantic.dimensions[:8]]
            parts = [metric_check.reason]
            if numeric_cols:
                parts.append(f"Available numeric fields: {', '.join(f'`{col}`' for col in numeric_cols)}.")
            if categorical_cols:
                parts.append(f"Available grouping fields: {', '.join(f'`{col}`' for col in categorical_cols)}.")
            return _plan_output(
                question,
                " ".join(parts),
                plan,
                route,
                findings=[],
                limitations=[metric_check.reason],
            )
    if plan.metric and plan.dimension:
        grouping_check = validate_grouping(plan.metric, plan.dimension, df)
        if not grouping_check.valid:
            if grouping_check.suggested_metric and grouping_check.suggested_metric != plan.metric:
                plan = replace(plan, metric=grouping_check.suggested_metric, metric_source="validation_override")
            elif grouping_check.suggested_dimension and grouping_check.suggested_dimension != plan.dimension:
                plan = replace(plan, dimension=grouping_check.suggested_dimension, dimension_source="validation_override")
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
        .agg(total="sum", mean="mean", median="median", count="count")
        .reset_index()
    )
    value_key = {"sum": "total", "mean": "mean", "median": "median", "count": "count"}.get(plan.aggregation or "sum", "total")
    grouped = grouped.sort_values(value_key, ascending=plan.ranking_direction == "ascending")
    if grouped.empty:
        return None
    limit = int(plan.limit or 20)
    rows = grouped.head(limit).to_dict(orient="records")
    top = rows[:5]
    aggregation_label = {"sum": "total", "mean": "average", "median": "median", "count": "count"}.get(plan.aggregation or "sum", "total")
    leader_bits = ", ".join(f"`{row[plan.dimension]}` ({aggregation_label} {float(row[value_key]):.2f}, n={int(row['count'])})" for row in top)
    filter_text = _filter_text(plan)
    proxy_alias = next((item for item in plan.safe_aliases if item.alias_type == "metric_proxy" and item.resolved_to == plan.metric), None)
    if proxy_alias:
        alias_text = f"No explicit `{plan.metric_alias_used}` field exists. Using `{plan.metric}` as the closest operational proxy. "
    else:
        alias_text = f"`{plan.metric_alias_used}` is treated as `{plan.metric}`. " if plan.metric_alias_used and str(plan.metric_alias_used).casefold() != str(plan.metric).casefold() else ""
    summary = f"{alias_text}{filter_text}`{plan.metric}` by `{plan.dimension}` is led by {leader_bits} using {aggregation_label} `{plan.metric}`."
    chart_rows = grouped.head(limit)[[plan.dimension, "total", "mean", "median", "count"]].to_dict(orient="records")
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
                    "ranking_scope": f"Top {min(len(chart_rows), limit)} `{plan.dimension}` groups by {aggregation_label} `{plan.metric}`.",
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
                    "top_n": int(min(len(chart_rows), limit)),
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
        if counts:
            min_count = min(counts)
            median_count = float(pd.Series(counts).median())
            max_count = max(counts)
            imbalance_ratio = max_count / max(min_count, 1)
            if (imbalance_ratio <= 1.25 or max_count - min_count <= 1) and min_count >= median_count * 0.5:
                return (f"No `{derived_field}` bins are materially sparse; the quantile binning produced balanced groups with {min_count} to {max_count} records per bin.", "bins_sparsity_balanced")
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
    dataset_registry: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    alternatives = [metric for metric in available_metrics if metric]
    suggestion = (
        " You can build a chart using "
        + ", ".join(f"`{metric}`" for metric in alternatives[:5])
        + " instead."
        if alternatives
        else " I do not see a reliable numeric metric in the available schema for that substitution."
    )
    # Check whether another loaded dataset contains the requested metric
    other_dataset_note = _other_dataset_has_field(requested_metric, dataset_registry)
    if other_dataset_note:
        summary = other_dataset_note + suggestion
    else:
        summary = (
            f"I cannot build a chart for `{requested_metric}`: `{source_name}` does not contain that field. "
            "I will not silently substitute a different metric."
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


def _detect_off_topic_question(question: str, df: pd.DataFrame, plan: Any) -> str:
    """Detect when the question's subject is entirely unrelated to the dataset.

    Returns a warning message if off-topic, or empty string if the question
    plausibly relates to the data.
    """
    _DOMAIN_NOUN_GROUPS: dict[str, tuple[str, ...]] = {
        "entertainment": ("film", "movie", "фильм", "кино", "genre", "жанр", "actor", "актер", "director", "режиссер", "tv show", "сериал", "episode", "эпизод"),
        "food": ("food", "еда", "блюд", "кухн", "cuisine", "dish", "recipe", "рецепт", "meal", "ingredient", "ингредиент"),
        "medical": ("patient", "пациент", "diagnosis", "диагноз", "treatment", "лечение", "symptom", "симптом", "doctor", "врач", "hospital", "больниц"),
        "education": ("student", "студент", "university", "университет", "teacher", "учител", "exam", "экзамен"),
        "sports": ("player", "игрок", "league", "лига", "матч", "goalkeeper", "вратар"),
        "music": ("song", "песн", "album", "альбом", "playlist", "плейлист"),
    }

    # Analytical terms that should NOT trigger off-topic detection even if
    # they are substrings of domain nouns (e.g., "seasonality" contains "season").
    _ANALYTICAL_SAFE = ("seasonality", "seasonal", "trending", "scoring", "distribution", "correlation", "cluster")

    q_lower = question.lower().replace("_", " ")
    col_text = " ".join(str(c).lower().replace("_", " ") for c in df.columns)

    matched_domain = ""
    matched_nouns: list[str] = []

    for domain, nouns in _DOMAIN_NOUN_GROUPS.items():
        hits = [n for n in nouns if _contains_domain_phrase(q_lower, n)]
        if hits:
            # Check that the match is not a substring of an analytical term
            real_hits = [
                n for n in hits
                if not any(safe in q_lower for safe in _ANALYTICAL_SAFE if n in safe and n != safe)
            ]
            if real_hits:
                matched_domain = domain
                matched_nouns = real_hits
                break

    if not matched_domain:
        return ""

    # Check if ANY of the domain nouns appear in column names
    if any(_contains_domain_phrase(col_text, noun) for noun in matched_nouns):
        return ""

    # Check if ANY domain nouns resolve via semantic alias to an existing column
    for noun in matched_nouns:
        if resolve_semantic_alias(noun, df) is not None:
            return ""

    # Also check broader domain vocabulary in column names
    domain_nouns = _DOMAIN_NOUN_GROUPS[matched_domain]
    if any(_contains_domain_phrase(col_text, noun) for noun in domain_nouns):
        return ""

    # Check column VALUES because domain signals may live in category values,
    # not only in column names.
    try:
        value_sample_parts: list[str] = []
        for col in df.select_dtypes(include="object").columns[:10]:
            unique_vals = df[col].dropna().unique()[:20]
            value_sample_parts.extend(str(v).lower() for v in unique_vals)
        value_text = " ".join(value_sample_parts)
        if any(_contains_domain_phrase(value_text, noun) for noun in matched_nouns):
            return ""
        if any(_contains_domain_phrase(value_text, noun) for noun in domain_nouns):
            return ""
    except Exception:
        pass

    # Check if the plan used user-mentioned columns or purely fell back to defaults
    if plan.metric and plan.metric.lower() in q_lower:
        return ""
    if plan.dimension and plan.dimension.lower() in q_lower:
        return ""

    noun_display = ", ".join(f"«{n}»" for n in matched_nouns[:3])
    return (
        f"The current dataset does not contain fields related to {noun_display}. "
        f"The question appears to be about {matched_domain} data, but the loaded dataset has different content."
    )


def _contains_domain_phrase(text: str, phrase: str) -> bool:
    """Match domain words as tokens/phrases so generic words like 'factor' do not match narrower nouns."""
    phrase = str(phrase or "").strip().lower()
    if not phrase:
        return False
    if re.search(r"[^\w\s]", phrase):
        return phrase in text
    pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
    return re.search(pattern, text) is not None


def _other_dataset_has_field(field_name: str, registry: list[dict[str, Any]] | None) -> str:
    """Check if any other loaded dataset contains the requested field.

    Returns a helpful message if found, or empty string if not.
    """
    if not registry:
        return ""
    field_lower = field_name.lower().replace("_", " ").strip()
    for entry in registry:
        columns = entry.get("column_names") or []
        for col in columns:
            if col.lower().replace("_", " ").strip() == field_lower:
                dataset_name = entry.get("display_name") or entry.get("source_name") or entry.get("dataset_id") or "another dataset"
                return (
                    f"I found `{col}` in the `{dataset_name}` dataset, but the current analysis is running against a different dataset. "
                    f"Switch to `{dataset_name}` or reference it explicitly to analyze `{col}`."
                )
    return ""


def _hard_stop_missing_fields(question: str, df: pd.DataFrame, plan: AuthoritativeQueryPlan) -> dict[str, Any]:
    """HARD STOP: refuse execution when explicitly requested fields/values are unavailable."""

    semantic = build_semantic_dataset_profile(df=df)
    numeric_cols = [str(col) for col in df.select_dtypes(include="number").columns if not _looks_identifier_like(df[col], str(col))]
    categorical_cols = [str(col) for col in df.columns if col not in set(numeric_cols)]
    missing_labels = ", ".join(f"`{field}`" for field in plan.missing_required_fields)
    parts = [
        f"I cannot execute this analysis because the current dataset does not contain the required {missing_labels}.",
    ]
    if plan.metric and plan.metric not in df.columns:
        parts.append(f"There is no `{plan.metric}` field in the loaded data.")
    for filt in plan.filters:
        col = filt.column
        if col not in df.columns:
            parts.append(f"There is no `{col}` field to filter by `{filt.value}`.")
        elif filt.value and str(filt.value) not in df[col].astype(str).values:
            parts.append(f"The value `{filt.value}` does not exist in `{col}`.")
    if numeric_cols:
        parts.append(f"Available numeric fields: {', '.join(f'`{col}`' for col in numeric_cols[:8])}.")
    if categorical_cols:
        parts.append(f"Available categorical fields: {', '.join(f'`{col}`' for col in categorical_cols[:8])}.")
    summary = " ".join(parts)
    return _output(
        question=question,
        summary=summary,
        findings=[],
        evidence=[f"Checked requested fields against {len(df.columns)} available columns."],
        limitations=[f"The explicitly requested {missing_labels} could not be resolved in the current schema."],
        next_steps=["Rephrase the question using available field names or upload a dataset with the requested fields."],
        code="",
        result_preview="",
        timeline=_timeline("hard_stop_missing_fields", missing=plan.missing_required_fields),
        artifacts=[],
        trace_metadata={
            "fallback": "hard_stop_missing_fields",
            "analysis_type": "hard_stop",
            "missing_required_fields": plan.missing_required_fields,
            "query_plan": plan.to_payload(),
            "suppress_key_findings": True,
        },
    )


def _insufficient_dataset_scope_response(question: str, df: pd.DataFrame, intent: Any, loaded_count: int) -> dict[str, Any]:
    """Prerequisite failure: multi-dataset intent with fewer than 2 datasets loaded."""

    intent_str = str(intent)
    intent_label = title_for_intent(intent) if hasattr(intent, "value") else "Multi-dataset analysis"

    if "joinability" in intent_str:
        summary = (
            f"I currently see only {loaded_count} dataset loaded, so there is nothing to join yet. "
            f"Joinability analysis requires at least two datasets — upload or attach a second dataset and ask again."
        )
    elif "warehouse" in intent_str:
        summary = (
            f"Warehouse design requires multiple datasets to identify shared keys, fact tables, and dimension tables. "
            f"Only {loaded_count} dataset is currently loaded. Load at least two datasets to proceed."
        )
    elif "schema_comparison" in intent_str or "semantic_reasoning" in intent_str:
        summary = (
            f"Comparing or reasoning across datasets requires at least two loaded datasets. "
            f"Only {loaded_count} dataset is available right now. Load a second dataset and try again."
        )
    elif "missing_links" in intent_str:
        summary = (
            f"Identifying missing links requires comparing entities across multiple datasets. "
            f"Only {loaded_count} dataset is currently loaded. Please load at least two datasets."
        )
    else:
        summary = (
            f"This question requires multiple datasets, but only {loaded_count} is currently loaded. "
            f"Load at least two datasets and ask again."
        )

    return _output(
        question=question,
        summary=summary,
        findings=[],
        evidence=[f"Intent classified as `{intent_str}` — requires at least 2 datasets, found {loaded_count}."],
        limitations=[f"Multi-dataset analysis cannot be performed with only {loaded_count} dataset."],
        next_steps=["Upload or attach a second dataset, then repeat this question."],
        code="",
        result_preview="",
        timeline=_timeline("insufficient_dataset_scope", intent=intent_str, loaded=loaded_count),
        artifacts=[],
        trace_metadata={
            "fallback": "insufficient_dataset_scope",
            "analysis_type": "insufficient_dataset_scope",
            "dataset_scope": "multi_dataset_required",
            "loaded_dataset_count": loaded_count,
            "suppress_key_findings": True,
        },
    )


def _global_intent_response(question: str, df: pd.DataFrame, routing: Any) -> dict[str, Any]:
    """Conceptual response for global/multi-dataset intents — produces findings and visual artifacts."""

    intent_label = title_for_intent(routing.question_intent_type) if hasattr(routing, "question_intent_type") else "Global analysis"
    semantic = build_semantic_dataset_profile(df=df)
    numeric_cols = [item.name for item in semantic.metrics[:6]] if hasattr(semantic, "metrics") else [str(col) for col in df.select_dtypes(include="number").columns[:6]]
    categorical_cols = [item.name for item in semantic.dimensions[:6]] if hasattr(semantic, "dimensions") else [str(col) for col in df.select_dtypes(exclude="number").columns[:6]]
    timestamp_cols = [item.name for item in semantic.timestamps[:3]] if hasattr(semantic, "timestamps") else []
    intent_type = str(getattr(routing, "question_intent_type", ""))

    findings: list[str] = []
    artifacts: list[dict[str, Any]] = []

    if "joinability" in intent_type:
        summary = (
            f"This is a **joinability** question that requires reasoning about shared keys and entity relationships across multiple datasets. "
            f"The current single dataset has {len(df.columns)} columns ({len(numeric_cols)} numeric, {len(categorical_cols)} categorical). "
            f"To assess joinability, load at least two datasets and compare their key fields."
        )
        findings.append(
            f"The loaded dataset contains {len(numeric_cols)} numeric fields and {len(categorical_cols)} categorical fields. "
            f"Joinability assessment requires a second dataset to compare shared keys and entity overlap."
        )
        # Compatibility heatmap: field-type distribution
        artifacts.append(_schema_profile_heatmap(df, numeric_cols, categorical_cols, timestamp_cols, title="Dataset Schema Profile"))

    elif "warehouse" in intent_type:
        facts = ", ".join(f"`{col}`" for col in numeric_cols[:4]) if numeric_cols else "none detected"
        dims = ", ".join(f"`{col}`" for col in categorical_cols[:4]) if categorical_cols else "none detected"
        summary = (
            f"**{intent_label}**: from this dataset, candidate fact measures are {facts} and candidate dimension fields are {dims}. "
            f"A warehouse design also needs entity relationships from additional tables."
        )
        findings.append(
            f"Candidate fact measures: {facts}. Candidate dimensions: {dims}. "
            f"A star schema requires at least one additional table to establish entity relationships."
        )
        # Table showing field classification
        artifacts.append(_field_classification_table(df, numeric_cols, categorical_cols, timestamp_cols))

    elif "missing_links" in intent_type:
        col_list = ", ".join(f"`{col}`" for col in list(df.columns)[:10])
        summary = (
            f"**{intent_label}**: to identify missing links, compare which entities and identifiers exist in each dataset. "
            f"Current dataset columns: {col_list}."
        )
        findings.append(
            f"The dataset exposes {len(df.columns)} fields. Identifying missing entity links requires a second dataset for cross-referencing."
        )
        artifacts.append(_schema_profile_heatmap(df, numeric_cols, categorical_cols, timestamp_cols, title="Entity Field Profile"))

    elif "schema_comparison" in intent_type or "semantic_reasoning" in intent_type:
        summary = (
            f"**{intent_label}**: this question requires conceptual or semantic comparison across multiple datasets. "
            f"The current dataset has {len(df)} rows and {len(df.columns)} columns "
            f"({len(numeric_cols)} numeric, {len(categorical_cols)} categorical)."
        )
        findings.append(
            f"Schema profile: {len(numeric_cols)} numeric fields, {len(categorical_cols)} categorical fields, "
            f"{len(timestamp_cols)} temporal fields across {len(df):,} rows. "
            f"Semantic comparison requires loading a second dataset."
        )
        artifacts.append(_schema_profile_heatmap(df, numeric_cols, categorical_cols, timestamp_cols, title="Schema Compatibility Profile"))

    elif any(marker in intent_type for marker in ("risk", "executive", "strategic", "customer", "hypothesis")):
        summary = (
            f"**{intent_label}**: this requires high-level reasoning across the dataset, not a single grouped comparison. "
            f"Key numeric fields: {', '.join(f'`{col}`' for col in numeric_cols[:4]) or 'none'}. "
            f"Key categorical fields: {', '.join(f'`{col}`' for col in categorical_cols[:4]) or 'none'}."
        )
        findings.append(
            f"High-level reasoning identifies {len(numeric_cols)} measurable metrics and {len(categorical_cols)} segmentation dimensions. "
            f"This dataset supports analytical exploration at the metric and segment level."
        )
    else:
        summary = (
            f"**{intent_label}**: this question requires conceptual or multi-dataset reasoning rather than a single grouped metric comparison. "
            f"The dataset has {len(df)} rows and {len(df.columns)} columns."
        )
        findings.append(
            f"The dataset has {len(df):,} rows and {len(df.columns)} columns. "
            f"Multi-dataset reasoning requires loading additional datasets for comparison."
        )

    return _output(
        question=question,
        summary=summary,
        findings=findings,
        evidence=[f"Intent classified as `{intent_type}` — grouped analysis suppressed."],
        limitations=["This answer is based on structural reasoning. Upload additional datasets or refine the question for deeper analysis."],
        next_steps=["Load additional datasets to compare schemas and key fields." if "multi" in intent_type or "join" in intent_type else "Ask a specific metric-level question for concrete analysis."],
        code="",
        result_preview="",
        timeline=_timeline("global_intent_bypass", intent=intent_type),
        artifacts=artifacts,
        trace_metadata={
            "fallback": "global_intent_bypass",
            "analysis_type": intent_type or "cross_dataset",
        },
    )


# ---------------------------------------------------------------------------
# Operation detection functions
# ---------------------------------------------------------------------------

def _is_temporal_shift_question(question: str) -> bool:
    """Detect questions about changes over time, before/after, shifts."""
    normalized = _normalize(question)
    markers = (
        "change after", "changed after", "change since", "changed since",
        "before and after", "before/after", "shift", "shifted",
        "evolution", "evolved", "trend", "trends", "hidden trend",
        "growing fastest", "fastest growing", "growth",
        "expansion", "strongest period", "strongest expansion",
        "изменил", "после", "до и после", "тренд",
    )
    return any(marker in normalized for marker in markers)


def _is_multi_factor_question(question: str) -> bool:
    """Detect questions about multiple factors, risk factors, drivers."""
    normalized = _normalize(question)
    factor_markers = (
        "which factor", "risk factor", "factors associated", "most associated",
        "predictors of", "drivers of", "what predicts", "what drives",
        "which variables", "which indicators", "strongest predictor",
        "most important factor", "combination of risk", "combinations of risk",
        "which features", "most correlated with",
        "какие фактор", "факторы риска",
    )
    return any(marker in normalized for marker in factor_markers)


def _is_growth_question(question: str) -> bool:
    """Detect questions about growth, fastest-growing, trending up."""
    normalized = _normalize(question)
    growth_markers = (
        "growing fastest", "fastest growing", "fastest-growing",
        "increasing most", "trending up", "growth rate",
        "gaining popularity", "rising fastest",
        "растет быстрее", "быстрее всего",
    )
    return any(marker in normalized for marker in growth_markers)


def _is_visual_request(question: str) -> bool:
    """Detect if the user explicitly asks for a visual output."""
    normalized = _normalize(question)
    return any(marker in normalized for marker in (
        "chart", "visualiz", "plot", "graph", "histogram",
        "show distribution", "build a ", "create a ",
        "compare visually", "visual analysis",
        "график", "визуализ", "диаграмм", "гистограмм",
        "построй", "покажи распределение",
    ))


def _is_diversity_question(question: str) -> bool:
    """Detect questions about diversity, variety, richness."""
    normalized = _normalize(question)
    markers = (
        "diversity", "diverse", "variety", "richness",
        "how varied", "how diverse", "разнообразие", "разнообраз",
    )
    return any(marker in normalized for marker in markers)


def _is_duration_by_category_question(question: str) -> bool:
    """Detect questions about duration/length grouped by category."""
    normalized = _normalize(question)
    duration_markers = ("duration", "length", "runtime", "длительност", "продолжительност")
    category_markers = ("by genre", "by category", "by type", "by group", "by diagnosis",
                        "по жанр", "по категор", "по тип")
    has_duration = any(m in normalized for m in duration_markers)
    has_category = any(m in normalized for m in category_markers)
    has_average = any(m in normalized for m in ("average", "mean", "median", "средн"))
    return (has_duration and has_category) or (has_duration and has_average)


def _is_strategic_synthesis_question(question: str) -> bool:
    """Detect questions asking for strategic synthesis or executive insights."""
    normalized = _normalize(question)
    strategic_markers = (
        "strategically important", "strategic importance", "strategy",
        "business insight", "executive", "key insight", "important finding",
        "important findings", "policymaker", "policy", "board", "leadership",
        "what can we learn", "main risk", "main risks", "public health risk",
        "business risk", "operational risk", "student performance risk",
        "fraud risk", "ecological risk", "risk summary",
        "стратегически", "стратег", "ключевые выводы",
    )
    if any(marker in normalized for marker in strategic_markers):
        return True
    return "summarize" in normalized and "risk" in normalized


# ---------------------------------------------------------------------------
# Temporal category mix shift analysis
# ---------------------------------------------------------------------------

def _temporal_mix_shift_response(
    question: str,
    df: pd.DataFrame,
    category_col: str,
    year_col: str,
) -> dict[str, Any]:
    """Compute before/after distribution shift for a category field over time.

    Answers: 'How did content strategy change after 2018?'
    """
    language = ResponseLanguagePolicy.from_message(question, protected_terms=df.columns)
    # Extract threshold year from question
    import re
    year_match = re.search(r'\b(19|20)\d{2}\b', question)
    if year_match:
        threshold = int(year_match.group())
    else:
        threshold = int(df[year_col].median())

    # Multi-label explode if needed
    delimiter = detect_multi_label_column(df[category_col]) if category_col in df.columns else None
    working = explode_multi_label(df, category_col, delimiter) if delimiter else df

    before = working[working[year_col] <= threshold]
    after = working[working[year_col] > threshold]

    before_counts = before[category_col].value_counts().head(10)
    after_counts = after[category_col].value_counts().head(10)

    all_categories = sorted(set(before_counts.index) | set(after_counts.index))
    shift_data = []
    for cat in all_categories:
        b = int(before_counts.get(cat, 0))
        a = int(after_counts.get(cat, 0))
        change = a - b
        shift_data.append({"category": cat, "before": b, "after": a, "change": change})
    shift_data.sort(key=lambda x: x["change"], reverse=True)

    emerging = [d for d in shift_data if d["change"] > 0]
    declining = [d for d in shift_data if d["change"] < 0]

    emerging_text = ", ".join(f"`{d['category']}` (+{d['change']})" for d in emerging[:3]) or "none"
    declining_text = ", ".join(f"`{d['category']}` ({d['change']})" for d in declining[:3]) or "none"

    summary = (
        f"Category distribution shift for `{category_col}` around {threshold}: "
        f"Before ≤{threshold}: {len(before)} records. After >{threshold}: {len(after)} records. "
        f"Emerging categories: {emerging_text}. "
        f"Declining categories: {declining_text}."
    )

    chart_rows = [{"category": d["category"], "before": d["before"], "after": d["after"]} for d in shift_data[:15]]
    findings = [
        summary,
        f"The dataset shows {len(emerging)} categories growing and {len(declining)} declining after {threshold}.",
    ]

    return _output(
        question=question,
        summary=summary,
        findings=findings,
        evidence=[f"Compared `{category_col}` distribution before and after {threshold} using `{year_col}`."],
        limitations=["This measures count-based distribution shift, not a causal change."],
        next_steps=[
            f"Examine the fastest-growing categories in detail.",
            f"Check if declining categories are being replaced or are seasonal.",
        ],
        code=f"before = df[df[{year_col!r}] <= {threshold}]; after = df[df[{year_col!r}] > {threshold}]",
        result_preview=pd.DataFrame(chart_rows).to_string(index=False) if chart_rows else "",
        timeline=_timeline("category_mix_shift", category=category_col, year=year_col, threshold=threshold),
        artifacts=[
            {
                "artifact_type": "chart",
                "title": f"Category shift around {threshold}",
                "content": {"chart_type": "bar", "x": "category", "y": "after", "rows": chart_rows},
                "visibility": "user",
                "pinned": True,
                "metadata": {"source": "category_mix_shift"},
            },
            {
                "artifact_type": "table",
                "title": f"Before/after comparison ({threshold})",
                "content": pd.DataFrame(chart_rows).to_string(index=False) if chart_rows else "",
                "visibility": "user",
                "pinned": False,
                "metadata": {"source": "category_mix_shift"},
            },
        ],
        trace_metadata={
            "fallback": "category_mix_shift",
            "analysis_type": "category_mix_shift",
            "dimension": category_col,
            "metric": "record_count",
            "temporal_field": year_col,
            "threshold": threshold,
        },
    )


# ---------------------------------------------------------------------------
# Diversity analysis
# ---------------------------------------------------------------------------

def _diversity_analysis_response(
    question: str,
    df: pd.DataFrame,
    group_col: str,
    entity_col: str,
) -> dict[str, Any]:
    """Compute diversity/richness of entity_col across groups in group_col.

    Answers: 'Compare regional content diversity'
    """
    language = ResponseLanguagePolicy.from_message(question, protected_terms=df.columns)
    # Multi-label explode if needed
    delimiter = detect_multi_label_column(df[entity_col]) if entity_col in df.columns else None
    working = explode_multi_label(df, entity_col, delimiter) if delimiter else df

    diversity_rows = []
    for group_val, group_df in working.groupby(group_col, dropna=False):
        unique_count = group_df[entity_col].nunique()
        total_count = len(group_df)
        top_entity = group_df[entity_col].value_counts().head(1)
        top_name = str(top_entity.index[0]) if len(top_entity) > 0 else ""
        top_share = float(top_entity.iloc[0] / max(total_count, 1)) if len(top_entity) > 0 else 0
        diversity_rows.append({
            "group": str(group_val),
            "unique_count": unique_count,
            "total_records": total_count,
            "top_entity": top_name,
            "concentration": round(top_share, 3),
        })

    diversity_rows.sort(key=lambda x: x["unique_count"], reverse=True)
    most_diverse = diversity_rows[0] if diversity_rows else None
    least_diverse = diversity_rows[-1] if diversity_rows else None

    summary = (
        f"Diversity of `{entity_col}` across `{group_col}`: "
        f"Most diverse: `{most_diverse['group']}` ({most_diverse['unique_count']} unique). "
        f"Least diverse: `{least_diverse['group']}` ({least_diverse['unique_count']} unique). "
        f"Across {len(diversity_rows)} groups."
    ) if most_diverse else f"No groups found for diversity analysis."

    findings = [
        summary,
        f"Concentration varies: `{most_diverse['group']}` top entity is `{most_diverse['top_entity']}` ({most_diverse['concentration']:.0%}), "
        f"while `{least_diverse['group']}` is dominated by `{least_diverse['top_entity']}` ({least_diverse['concentration']:.0%})."
        if most_diverse and least_diverse else "",
    ]
    findings = [f for f in findings if f]

    chart_rows = [{"group": r["group"], "unique_count": r["unique_count"], "concentration": r["concentration"]} for r in diversity_rows[:20]]

    return _output(
        question=question,
        summary=summary,
        findings=findings,
        evidence=[f"Computed unique `{entity_col}` count and concentration per `{group_col}`."],
        limitations=["Diversity is measured by unique count; entropy-based measures may reveal more nuance."],
        next_steps=[
            f"Examine the most concentrated groups to understand specialization.",
            f"Compare diversity across time periods if a temporal field exists.",
        ],
        code=f"diversity = df.groupby({group_col!r})[{entity_col!r}].nunique().sort_values(ascending=False)",
        result_preview=pd.DataFrame(chart_rows).to_string(index=False) if chart_rows else "",
        timeline=_timeline("diversity_analysis", group=group_col, entity=entity_col),
        artifacts=[
            {
                "artifact_type": "chart",
                "title": f"{entity_col} diversity by {group_col}",
                "content": {"chart_type": "bar", "x": "group", "y": "unique_count", "rows": chart_rows},
                "visibility": "user",
                "pinned": True,
                "metadata": {"source": "diversity_analysis"},
            },
        ],
        trace_metadata={
            "fallback": "diversity_analysis",
            "analysis_type": "diversity_analysis",
            "dimension": group_col,
            "metric": f"unique_{entity_col}_count",
        },
    )


# ---------------------------------------------------------------------------
# Duration-by-category analysis
# ---------------------------------------------------------------------------

def _duration_by_category_response(
    question: str,
    df: pd.DataFrame,
    duration_col: str,
    category_col: str,
) -> dict[str, Any]:
    """Parse duration text and compute average by category.

    Answers: 'Compare average durations by genre'
    """
    language = ResponseLanguagePolicy.from_message(question, protected_terms=df.columns)
    parsed = _parse_duration_series(df[duration_col])
    if parsed is None:
        return _output(
            question=question,
            summary=f"Could not parse `{duration_col}` as numeric duration values.",
            findings=[],
            evidence=[],
            limitations=[f"`{duration_col}` values could not be converted to comparable numeric durations."],
            next_steps=["Check the format of the duration column and ensure it contains parseable values."],
            code="",
            result_preview="",
            timeline=_timeline("duration_parse_failure"),
            trace_metadata={"fallback": "duration_parse_failure", "analysis_type": "fallback"},
        )

    working = df.copy()
    working["_parsed_duration"] = parsed
    valid = working.dropna(subset=["_parsed_duration"])

    # Multi-label explode if needed
    delimiter = detect_multi_label_column(valid[category_col]) if category_col in valid.columns else None
    if delimiter:
        valid = explode_multi_label(valid, category_col, delimiter)

    grouped = (
        valid.groupby(category_col, dropna=False)["_parsed_duration"]
        .agg(["mean", "median", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    grouped.columns = [category_col, "avg_duration", "median_duration", "count"]
    grouped["avg_duration"] = grouped["avg_duration"].round(1)
    grouped["median_duration"] = grouped["median_duration"].round(1)

    top = grouped.head(3)
    top_text = ", ".join(
        f"`{row[category_col]}` ({row['avg_duration']:.0f} min, n={int(row['count'])})"
        for _, row in top.iterrows()
    )

    # Check for mixed units warning
    has_seasons = df[duration_col].astype(str).str.contains("season", case=False, na=False).any()
    has_minutes = df[duration_col].astype(str).str.contains("min", case=False, na=False).any()
    mixed_units = has_seasons and has_minutes

    summary = (
        f"Average parsed duration by `{category_col}`: {top_text}. "
        f"Analyzed {len(valid)} records with parseable duration values."
    )
    if mixed_units:
        summary += " Note: mixed units detected (minutes and seasons) — results combine both."

    findings = [summary]
    if mixed_units:
        findings.append("Mixed duration units (minutes and seasons) are present. Consider filtering to one unit type for cleaner comparison.")

    chart_rows = grouped.head(20).to_dict(orient="records")

    return _output(
        question=question,
        summary=summary,
        findings=findings,
        evidence=[f"Parsed `{duration_col}` into numeric minutes and grouped by `{category_col}`."],
        limitations=["Duration parsing assumes '90 min' → 90 and '1 Season' → 1. Mixed units may affect averages." if mixed_units else "Duration parsing may not handle all formats."],
        next_steps=[
            "Filter to movies only (or TV shows only) for a cleaner comparison." if mixed_units else f"Explore the distribution within each `{category_col}` group.",
        ],
        code=f"grouped = df.groupby({category_col!r})['_parsed_duration'].agg(['mean', 'median', 'count'])",
        result_preview=grouped.head(15).to_string(index=False),
        timeline=_timeline("duration_by_category", duration=duration_col, category=category_col),
        artifacts=[
            {
                "artifact_type": "chart",
                "title": f"Average duration by {category_col}",
                "content": {"chart_type": "bar", "x": category_col, "y": "avg_duration", "rows": chart_rows},
                "visibility": "user",
                "pinned": True,
                "metadata": {"source": "duration_by_category"},
            },
        ],
        trace_metadata={
            "fallback": "duration_by_category",
            "analysis_type": "duration_by_category",
            "dimension": category_col,
            "metric": "parsed_duration",
        },
    )


# ---------------------------------------------------------------------------
# Multi-factor association analysis
# ---------------------------------------------------------------------------

def _multi_factor_association_response(question: str, df: pd.DataFrame) -> dict[str, Any]:
    """Rank all candidate factors by association strength with a target outcome."""
    from source.product.semantic_role_assignment import assign_semantic_roles
    language = ResponseLanguagePolicy.from_message(question, protected_terms=df.columns)
    roles = assign_semantic_roles(question, df)
    target_col = roles.target_variable if roles else None

    # If no target found, try to find a binary outcome column
    if not target_col:
        numeric_cols = [str(col) for col in df.select_dtypes(include="number").columns]
        binary_candidates = [
            col for col in numeric_cols
            if df[col].dropna().isin([0, 1]).all() and df[col].nunique() == 2
        ]
        q_lower = question.lower()
        for col in binary_candidates:
            col_words = col.lower().replace("_", " ").split()
            if any(w in q_lower for w in col_words if len(w) > 3):
                target_col = col
                break
        if not target_col and binary_candidates:
            target_col = binary_candidates[0]

    if not target_col or target_col not in df.columns:
        return _output(
            question=question,
            summary="Could not identify a clear target/outcome variable for multi-factor analysis. Specify which outcome to analyze.",
            findings=[],
            evidence=[],
            limitations=["No binary outcome or target variable could be identified from the question and dataset."],
            next_steps=["Specify the target variable, e.g., 'Which factors are associated with ColumnName?'"],
            code="", result_preview="",
            timeline=_timeline("multi_factor_limitation"),
            trace_metadata={"fallback": "multi_factor_limitation", "analysis_type": "fallback"},
        )

    associations: list[dict[str, Any]] = []
    for col in df.columns:
        col_name = str(col)
        if col_name == target_col or _looks_identifier_like(df[col_name], col_name):
            continue
        try:
            if pd.api.types.is_numeric_dtype(df[col_name]):
                if df[col_name].dropna().isin([0, 1]).all() and df[col_name].nunique() == 2:
                    group_rates = df.groupby(col_name)[target_col].mean()
                    if len(group_rates) == 2:
                        gap = abs(float(group_rates.iloc[1] - group_rates.iloc[0]))
                        rate_1 = float(group_rates.get(1, group_rates.iloc[-1]))
                        rate_0 = float(group_rates.get(0, group_rates.iloc[0]))
                        associations.append({"factor": col_name, "type": "binary", "gap": gap, "rate_exposed": rate_1, "rate_unexposed": rate_0, "n": int(df[col_name].notna().sum())})
                else:
                    valid = df[[col_name, target_col]].dropna()
                    if len(valid) > 10:
                        corr = abs(float(valid[col_name].corr(valid[target_col])))
                        if corr == corr:
                            associations.append({"factor": col_name, "type": "numeric", "gap": corr, "correlation": corr, "n": len(valid)})
            else:
                nunique = df[col_name].nunique()
                if 2 <= nunique <= 20:
                    group_rates = df.groupby(col_name)[target_col].mean()
                    if len(group_rates) >= 2:
                        gap = float(group_rates.max() - group_rates.min())
                        associations.append({"factor": col_name, "type": "categorical", "gap": gap, "top_group": str(group_rates.idxmax()), "top_rate": float(group_rates.max()), "n": int(df[col_name].notna().sum())})
        except Exception:
            continue

    associations.sort(key=lambda x: x["gap"], reverse=True)
    if not associations:
        return _output(
            question=question,
            summary=f"No factors with measurable association to `{target_col}` were found.",
            findings=[], evidence=[f"Checked {len(df.columns) - 1} candidate columns against `{target_col}`."],
            limitations=["No association found."],
            next_steps=["Examine the target variable distribution."],
            code="", result_preview="",
            timeline=_timeline("multi_factor_no_association"),
            trace_metadata={"fallback": "multi_factor_no_association", "analysis_type": "multi_factor_association"},
        )

    top_factors = associations[:5]
    overall_rate = float(df[target_col].mean())
    ranked_text = []
    for i, assoc in enumerate(top_factors, 1):
        factor = assoc["factor"]
        if assoc["type"] == "binary":
            ranked_text.append(f"{i}. `{factor}` (prevalence gap {assoc['gap']:.1%}): `{target_col}` rate is {assoc['rate_exposed']:.1%} when `{factor}`=1 vs {assoc['rate_unexposed']:.1%} when `{factor}`=0")
        elif assoc["type"] == "numeric":
            ranked_text.append(f"{i}. `{factor}` (correlation {assoc.get('correlation', assoc['gap']):.2f}): numeric association with `{target_col}`")
        else:
            ranked_text.append(f"{i}. `{factor}` (category gap {assoc['gap']:.1%}): highest `{target_col}` rate in `{assoc.get('top_group', '?')}` ({assoc.get('top_rate', 0):.1%})")
    findings = [
        f"Factors ranked by association strength with `{target_col}` (overall rate {overall_rate:.1%}):\n" + "\n".join(ranked_text),
        f"The strongest factor is `{top_factors[0]['factor']}` with a {top_factors[0]['gap']:.1%} gap. These are associations, not causal claims.",
    ]
    chart_rows = [{"factor": a["factor"], "association_strength": round(a["gap"], 3), "type": a["type"]} for a in top_factors]

    return _output(
        question=question,
        summary=" ".join(findings[:2]),
        findings=findings,
        evidence=[f"Computed association scores for {len(associations)} candidate factors against `{target_col}` across {len(df)} records."],
        limitations=["Association does not imply causation. Confounding variables may explain observed differences."],
        next_steps=[f"Investigate confounders by stratifying `{top_factors[0]['factor']}` by other variables.", "Check if associations hold across subgroups."],
        code="", result_preview=pd.DataFrame(chart_rows).to_string(index=False) if chart_rows else "",
        timeline=_timeline("multi_factor_association", target=target_col, factors=len(top_factors)),
        artifacts=[
            {"artifact_type": "chart", "title": f"Factor association with {target_col}", "content": {"chart_type": "bar", "x": "factor", "y": "association_strength", "rows": chart_rows}, "visibility": "user", "pinned": True, "metadata": {"source": "multi_factor_association"}},
            {"artifact_type": "table", "title": f"Factor rankings for {target_col}", "content": pd.DataFrame(chart_rows).to_string(index=False) if chart_rows else "", "visibility": "user", "pinned": False, "metadata": {"source": "multi_factor_association"}},
        ],
        trace_metadata={"fallback": "multi_factor_association", "analysis_type": "multi_factor_association", "target": target_col, "metric": target_col},
    )


# ---------------------------------------------------------------------------
# Category growth analysis
# ---------------------------------------------------------------------------

def _category_growth_response(
    question: str, df: pd.DataFrame, category_col: str, time_col: str,
) -> dict[str, Any]:
    """Compute growth rates for categories across time periods."""
    language = ResponseLanguagePolicy.from_message(question, protected_terms=df.columns)
    delimiter = detect_multi_label_column(df[category_col]) if category_col in df.columns else None
    working = explode_multi_label(df, category_col, delimiter) if delimiter else df

    time_values = working[time_col].dropna().sort_values()
    if len(time_values) < 2:
        return _output(
            question=question, summary=f"Not enough temporal variation in `{time_col}` to compute growth.",
            findings=[], evidence=[], limitations=[f"`{time_col}` has insufficient variation."],
            next_steps=[], code="", result_preview="",
            timeline=_timeline("growth_limitation"),
            trace_metadata={"fallback": "growth_limitation", "analysis_type": "fallback"},
        )
    median_time = time_values.median()
    early = working[working[time_col] <= median_time]
    recent = working[working[time_col] > median_time]
    early_counts = early[category_col].value_counts()
    recent_counts = recent[category_col].value_counts()
    all_cats = sorted(set(early_counts.index) | set(recent_counts.index))
    growth_data = []
    for cat in all_cats:
        e = int(early_counts.get(cat, 0))
        r = int(recent_counts.get(cat, 0))
        abs_growth = r - e
        pct_growth = ((r - e) / max(e, 1)) * 100 if e > 0 else (100.0 if r > 0 else 0.0)
        growth_data.append({"category": str(cat), "early_count": e, "recent_count": r, "absolute_growth": abs_growth, "growth_pct": round(pct_growth, 1)})
    growth_data.sort(key=lambda x: x["growth_pct"], reverse=True)
    fastest = growth_data[:3]
    declining = [d for d in growth_data if d["absolute_growth"] < 0][:3]
    fastest_text = ", ".join(f"`{d['category']}` (+{d['growth_pct']:.0f}%, {d['early_count']}→{d['recent_count']})" for d in fastest)
    declining_text = ", ".join(f"`{d['category']}` ({d['growth_pct']:.0f}%)" for d in declining) if declining else "none"
    summary = f"Growth analysis of `{category_col}` by `{time_col}` (split at {median_time}): Fastest growing: {fastest_text}. Declining: {declining_text}."
    findings = [
        summary,
        f"Out of {len(all_cats)} categories, {sum(1 for d in growth_data if d['absolute_growth'] > 0)} are growing and {sum(1 for d in growth_data if d['absolute_growth'] < 0)} are declining.",
    ]
    chart_rows = [{"category": d["category"], "growth_pct": d["growth_pct"], "recent_count": d["recent_count"]} for d in growth_data[:15]]
    return _output(
        question=question, summary=summary, findings=findings,
        evidence=[f"Compared `{category_col}` counts in early (≤{median_time}) vs recent (>{median_time}) periods using `{time_col}`."],
        limitations=["Growth is measured by record count change between periods, not continuous rate."],
        next_steps=["Examine the fastest-growing categories in detail.", "Check if growth reflects new entries or increasing activity."],
        code=f"early = df[df[{time_col!r}] <= {median_time}]; recent = df[df[{time_col!r}] > {median_time}]",
        result_preview=pd.DataFrame(chart_rows).to_string(index=False) if chart_rows else "",
        timeline=_timeline("category_growth", category=category_col, time=time_col),
        artifacts=[{"artifact_type": "chart", "title": f"{category_col} growth rates", "content": {"chart_type": "bar", "x": "category", "y": "growth_pct", "rows": chart_rows}, "visibility": "user", "pinned": True, "metadata": {"source": "category_growth"}}],
        trace_metadata={"fallback": "category_growth", "analysis_type": "category_growth", "dimension": category_col, "metric": "growth_rate", "temporal_field": time_col},
    )


# ---------------------------------------------------------------------------
# Chart artifact enforcement
# ---------------------------------------------------------------------------

def _ensure_chart_artifact(result: dict[str, Any], question: str) -> dict[str, Any]:
    """If the user asked for a visual but no chart artifact was produced, inject one from computed evidence."""
    if not _is_visual_request(question):
        return result
    artifacts = result.get("artifacts", [])
    has_chart = any(
        (a.get("artifact_type") == "chart" if isinstance(a, dict) else getattr(a, "artifact_type", None) == "chart")
        for a in artifacts
    )
    if has_chart:
        return result
    trace = result.get("trace_metadata", {})
    metric = trace.get("metric", "")
    dimension = trace.get("dimension", "")
    if not metric or not dimension:
        limitations = list(result.get("limitations", []))
        limitations.append("A visualization was requested but could not be generated because the metric or dimension could not be determined.")
        result["limitations"] = limitations
        return result
    chart_artifact = {
        "artifact_type": "chart",
        "title": f"{metric} by {dimension}",
        "content": {"chart_type": "bar", "x": dimension, "y": metric, "rows": []},
        "visibility": "user",
        "pinned": True,
        "metadata": {"source": "chart_enforcement", "analysis_type": trace.get("analysis_type", "")},
    }
    result_preview = result.get("result_preview", "")
    if result_preview and isinstance(result_preview, str) and len(result_preview) > 10:
        chart_artifact["content"]["preview"] = result_preview[:500]
    result.setdefault("artifacts", []).append(chart_artifact)
    return result


# ---------------------------------------------------------------------------
# Binary indicators, ordered prevalence, and factor diversity
# ---------------------------------------------------------------------------

_BINARY_PREVALENCE_MARKERS = (
    "most common", "most frequent", "most prevalent", "appear most common",
    "conditions", "condition", "indicators", "indicator", "symptoms", "symptom",
    "risk factors", "flags", "flag", "prevalent", "frequent",
)

_ORDERED_GROUP_MARKERS = (
    "age group", "age groups", "older", "younger", "ordered", "bracket",
    "band", "level", "levels", "severity", "grade level", "tenure",
    "income bracket", "elevation", "low to high", "increase", "decrease",
)

_FACTOR_DIVERSITY_MARKERS = (
    "factor diversity", "factors diversity", "risk-factor diversity",
    "risk factor diversity", "lifestyle-factor diversity",
    "lifestyle factor diversity", "feature mix", "behavior-factor diversity",
    "behavior factor diversity",
)


def _is_binary_indicator_prevalence_question(question: str) -> bool:
    normalized = _normalize(question)
    has_frequency = any(marker in normalized for marker in ("most common", "most frequent", "most prevalent", "prevalent", "frequent"))
    has_indicator = any(marker in normalized for marker in ("condition", "indicator", "symptom", "risk factor", "flag", "diagnosis", "outcome"))
    return has_frequency and has_indicator


def _is_ordered_group_prevalence_question(question: str) -> bool:
    normalized = _normalize(question)
    return (
        not _is_visual_request(question)
        and any(marker in normalized for marker in _ORDERED_GROUP_MARKERS)
        and any(marker in normalized for marker in ("risk", "rate", "prevalence", "incidence", "failure", "churn", "default", "fraud", "outcome"))
        and not _is_ordered_indicator_gradient_question(question)
    )


def _is_ordered_indicator_gradient_question(question: str) -> bool:
    normalized = _normalize(question)
    return (
        any(marker in normalized for marker in ("become more common", "becomes more common", "increase", "increases", "more common among older"))
        and any(marker in normalized for marker in ("indicator", "indicators", "condition", "conditions", "flag", "flags", "symptom", "symptoms", "risk factor"))
        and any(marker in normalized for marker in _ORDERED_GROUP_MARKERS)
    )


def _is_factor_diversity_question(question: str) -> bool:
    normalized = _normalize(question)
    return any(marker in normalized for marker in _FACTOR_DIVERSITY_MARKERS) or (
        "diversity" in normalized and any(marker in normalized for marker in ("factor", "feature mix", "behavior", "lifestyle", "risk"))
    )


def _binary_indicator_columns(df: pd.DataFrame, question: str = "") -> list[str]:
    """Return generic yes/no indicator fields, excluding IDs and administrative flags unless requested."""
    normalized = _normalize(question)
    include_admin = any(marker in normalized for marker in ("cost", "access", "administrative", "admin", "fee", "payment"))
    excluded_name_markers = (
        "id", "uuid", "code", "key", "zip", "postal",
        "gender", "sex", "male", "female", "age", "year", "month", "date",
        "segment", "group", "category", "type",
    )
    admin_markers = ("cost", "fee", "payment", "paid", "access", "doc", "doctor", "admin", "coverage")
    cols: list[str] = []
    for col in df.columns:
        name = str(col)
        norm = _normalize(name)
        if any(marker == norm or marker in norm for marker in excluded_name_markers):
            continue
        if not include_admin and any(marker in norm for marker in admin_markers):
            continue
        if _looks_identifier_like(df[col], name):
            continue
        converted = _binary_numeric_series(df[col])
        if converted is not None and converted.notna().sum() > 0:
            cols.append(name)
    return cols


def _binary_numeric_series(series: pd.Series) -> pd.Series | None:
    non_null = series.dropna()
    if non_null.empty:
        return None
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        unique = set(float(v) for v in numeric.dropna().unique())
        if unique and unique <= {0.0, 1.0}:
            return numeric
    values = series.astype(str).str.lower().str.strip()
    mapping = {
        "yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0,
        "y": 1, "n": 0, "t": 1, "f": 0,
        "positive": 1, "negative": 0, "present": 1, "absent": 0,
        "pass": 1, "fail": 0, "passed": 1, "failed": 0,
    }
    mapped = values.map(mapping)
    if mapped.notna().sum() == len(non_null) and set(mapped.dropna().unique()) <= {0, 1}:
        return mapped.astype(float)
    return None


def _binary_indicator_prevalence_response(question: str, df: pd.DataFrame) -> dict[str, Any] | None:
    cols = _binary_indicator_columns(df, question)
    if not cols:
        return None
    rows: list[dict[str, Any]] = []
    for col in cols:
        numeric = _binary_numeric_series(df[col])
        if numeric is None:
            continue
        total = int(numeric.notna().sum())
        positive = int(numeric.sum())
        prevalence = float(numeric.mean()) if total else 0.0
        rows.append({
            "indicator": col,
            "prevalence": round(prevalence, 4),
            "prevalence_pct": f"{prevalence:.1%}",
            "positive_count": positive,
            "total": total,
        })
    if not rows:
        return None
    rows.sort(key=lambda item: (-float(item["prevalence"]), str(item["indicator"])))
    top = rows[:8]
    top_text = ", ".join(f"`{row['indicator']}` ({row['prevalence_pct']}, {row['positive_count']}/{row['total']})" for row in top[:5])
    summary = f"The most prevalent binary indicators are {top_text}. These rankings use positive-share prevalence, not arbitrary numeric averages."
    chart_rows = [{"x": row["indicator"], "y": round(float(row["prevalence"]) * 100, 2)} for row in top]
    return _output(
        question=question,
        summary=summary,
        findings=[
            summary,
            f"Ranked {len(rows)} binary indicator fields by positive share across {len(df)} records.",
        ],
        evidence=[f"Computed positive counts and prevalence for {len(rows)} binary indicator columns."],
        limitations=["Administrative, access, cost, identifier, and demographic flags are excluded unless explicitly requested."],
        next_steps=["Break top indicators down by a relevant segment to see where prevalence concentrates."],
        code="# Rank binary indicator columns by positive share",
        result_preview=pd.DataFrame(top).to_string(index=False),
        timeline=_timeline("binary_indicator_prevalence", indicators=len(rows)),
        artifacts=[
            {"artifact_type": "chart", "title": "Binary indicator prevalence ranking", "content": {"chart_type": "bar", "x": "x", "y": "y", "metric": "prevalence (%)", "rows": chart_rows}, "visibility": "user", "pinned": True, "metadata": {"analysis_type": "binary_indicator_prevalence", "semantic_roles": {"indicators": cols}}},
            {"artifact_type": "table", "title": "Binary indicator prevalence", "content": top, "visibility": "user", "pinned": True, "metadata": {"analysis_type": "binary_indicator_prevalence", "semantic_roles": {"indicators": cols}}},
        ],
        trace_metadata={"fallback": "binary_indicator_prevalence", "analysis_type": "binary_indicator_prevalence", "operation": "BINARY_INDICATOR_PREVALENCE_RANKING", "indicator_count": len(rows), "metric": "prevalence"},
    )


def _ordered_group_column(question: str, df: pd.DataFrame) -> str | None:
    normalized = _normalize(question)
    ordered_markers = ("age", "tenure", "grade", "level", "severity", "income", "elevation", "risk band", "bracket", "band")
    for col in df.columns:
        col_norm = _normalize(str(col))
        if any(marker in normalized and marker in col_norm for marker in ordered_markers):
            if _is_orderable_series(df[col], str(col)):
                return str(col)
    for col in df.columns:
        name = str(col)
        if _is_orderable_series(df[col], name):
            return name
    return None


def _is_orderable_series(series: pd.Series, name: str) -> bool:
    norm = _normalize(name)
    if _looks_identifier_like(series, name) or "id" in norm or "key" in norm or "code" in norm:
        return False
    if _binary_numeric_series(series) is not None:
        return False
    if pd.api.types.is_numeric_dtype(series):
        return int(series.nunique(dropna=True)) >= 3
    values = [str(v) for v in series.dropna().unique()[:50]]
    return len(values) >= 3 and all(re.search(r"\d+", value) for value in values)


def _ordered_groups(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        unique = int(numeric.nunique(dropna=True))
        if unique > 6:
            ranked = numeric.rank(method="first")
            try:
                binned = pd.qcut(ranked, q=min(5, unique), duplicates="drop")
                labels = binned.map(lambda interval: f"{numeric[binned == interval].min():g}-{numeric[binned == interval].max():g}" if pd.notna(interval) else "Missing")
                return labels.astype("object")
            except Exception:
                pass
        return numeric.astype("object")
    return series.astype("object")


def _group_sort_key(value: Any) -> tuple[float, str]:
    text = str(value)
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        try:
            return float(numbers[0]), text
        except ValueError:
            pass
    return 0.0, text


def _ordered_group_prevalence_response(question: str, df: pd.DataFrame) -> dict[str, Any] | None:
    from source.product.semantic_role_assignment import assign_semantic_roles

    roles = assign_semantic_roles(question, df)
    if not roles or not roles.target_variable:
        return None
    target = roles.target_variable
    target_series = _binary_numeric_series(df[target])
    if target_series is None:
        return None
    group_col = _ordered_group_column(question, df)
    if not group_col or group_col == target:
        return None
    groups = _ordered_groups(df[group_col])
    working = pd.DataFrame({"group": groups, "_target": target_series}).dropna()
    if working.empty or working["group"].nunique() < 2:
        return None
    grouped = working.groupby("group")["_target"].agg(["mean", "sum", "count"]).reset_index()
    grouped["_sort"] = grouped["group"].map(_group_sort_key)
    grouped = grouped.sort_values("_sort")
    rows = []
    for _, row in grouped.iterrows():
        prevalence = float(row["mean"])
        rows.append({"group": str(row["group"]), "prevalence": round(prevalence, 4), "prevalence_pct": f"{prevalence:.1%}", "positive_count": int(row["sum"]), "total": int(row["count"])})
    change = float(rows[-1]["prevalence"]) - float(rows[0]["prevalence"])
    pair_changes = [
        (float(rows[i + 1]["prevalence"]) - float(rows[i]["prevalence"]), rows[i]["group"], rows[i + 1]["group"])
        for i in range(len(rows) - 1)
    ]
    strongest = max(pair_changes, key=lambda item: item[0])
    direction = "increases" if change > 0 else "decreases" if change < 0 else "is flat"
    summary = (
        f"`{target}` prevalence {direction} from {rows[0]['prevalence_pct']} in the lowest `{group_col}` group "
        f"to {rows[-1]['prevalence_pct']} in the highest group (change {change:.1%}). "
        f"The strongest adjacent increase is from `{strongest[1]}` to `{strongest[2]}` ({strongest[0]:.1%})."
    )
    chart_rows = [{"x": row["group"], "y": round(float(row["prevalence"]) * 100, 2)} for row in rows]
    return _output(
        question=question,
        summary=summary,
        findings=[summary, "This is ordered demographic or ordinal segmentation, not time-series forecasting."],
        evidence=[f"Computed `{target}` prevalence by ordered `{group_col}` groups across {len(working)} records."],
        limitations=["Observed prevalence gradients are associations and may be confounded by other variables."],
        next_steps=["Repeat the ordered-group comparison within another segment to check robustness."],
        code=f"# mean({target!r}) by ordered {group_col!r} groups",
        result_preview=pd.DataFrame(rows).to_string(index=False),
        timeline=_timeline("ordered_group_prevalence", target=target, group=group_col),
        artifacts=[
            {"artifact_type": "chart", "title": f"{target} prevalence by ordered {group_col}", "content": {"chart_type": "bar", "x": "x", "y": "y", "metric": "prevalence (%)", "rows": chart_rows}, "visibility": "user", "pinned": True, "metadata": {"analysis_type": "ordered_group_prevalence", "target": target, "grouping": group_col, "semantic_roles": {"target": target, "ordered_group": group_col}}},
            {"artifact_type": "table", "title": f"{target} prevalence by {group_col}", "content": rows, "visibility": "user", "pinned": True, "metadata": {"analysis_type": "ordered_group_prevalence", "target": target, "grouping": group_col}},
        ],
        trace_metadata={"fallback": "ordered_group_prevalence", "analysis_type": "ordered_group_prevalence", "operation": "ORDERED_GROUP_PREVALENCE_ANALYSIS", "target_variable": target, "grouping_variable": group_col, "metric": target, "dimension": group_col},
    )


def _ordered_indicator_gradient_response(question: str, df: pd.DataFrame) -> dict[str, Any] | None:
    group_col = _ordered_group_column(question, df)
    indicators = [col for col in _binary_indicator_columns(df, question) if col != group_col]
    if not group_col or not indicators:
        return None
    groups = _ordered_groups(df[group_col])
    rows: list[dict[str, Any]] = []
    for indicator in indicators:
        numeric = _binary_numeric_series(df[indicator])
        if numeric is None:
            continue
        working = pd.DataFrame({"group": groups, "_value": numeric}).dropna()
        grouped = working.groupby("group")["_value"].mean().reset_index()
        if grouped["group"].nunique() < 2:
            continue
        grouped["_sort"] = grouped["group"].map(_group_sort_key)
        grouped = grouped.sort_values("_sort")
        first = float(grouped["_value"].iloc[0])
        last = float(grouped["_value"].iloc[-1])
        rows.append({
            "indicator": indicator,
            "lowest_group_prevalence": round(first, 4),
            "highest_group_prevalence": round(last, 4),
            "increase": round(last - first, 4),
            "increase_pct": f"{last - first:.1%}",
        })
    if not rows:
        return None
    rows.sort(key=lambda item: (-float(item["increase"]), str(item["indicator"])))
    top = rows[:8]
    top_text = ", ".join(f"`{row['indicator']}` ({row['increase_pct']})" for row in top[:5])
    summary = f"Indicators with the largest increase across ordered `{group_col}` groups are {top_text}. The ranking compares prevalence in the lowest versus highest ordered group."
    chart_rows = [{"x": row["indicator"], "y": round(float(row["increase"]) * 100, 2)} for row in top]
    return _output(
        question=question,
        summary=summary,
        findings=[summary, f"Ranked {len(rows)} binary indicators by ordered-group prevalence gradient."],
        evidence=[f"Computed low-to-high prevalence changes by `{group_col}` for {len(rows)} binary indicators."],
        limitations=["A monotonic gradient is descriptive evidence, not proof of causation."],
        next_steps=["Inspect the top indicators by additional demographic or operational segments."],
        code=f"# prevalence gradient by ordered {group_col!r}",
        result_preview=pd.DataFrame(top).to_string(index=False),
        timeline=_timeline("ordered_indicator_gradient", group=group_col, indicators=len(rows)),
        artifacts=[
            {"artifact_type": "chart", "title": f"Indicator prevalence increase by {group_col}", "content": {"chart_type": "bar", "x": "x", "y": "y", "metric": "increase in prevalence (%)", "rows": chart_rows}, "visibility": "user", "pinned": True, "metadata": {"analysis_type": "ordered_indicator_gradient", "grouping": group_col, "semantic_roles": {"ordered_group": group_col, "indicators": indicators}}},
            {"artifact_type": "table", "title": f"Indicator gradients by {group_col}", "content": top, "visibility": "user", "pinned": True, "metadata": {"analysis_type": "ordered_indicator_gradient", "grouping": group_col}},
        ],
        trace_metadata={"fallback": "ordered_indicator_gradient", "analysis_type": "ordered_indicator_gradient", "operation": "ORDERED_INDICATOR_PREVALENCE_RANKING", "grouping_variable": group_col, "metric": "prevalence_gradient", "dimension": group_col},
    )


def _factor_diversity_response(question: str, df: pd.DataFrame) -> dict[str, Any] | None:
    factors = _binary_indicator_columns(df, question)
    if len(factors) < 2:
        return None
    group_col = _find_group_col_from_question(question, df, exclude=None)
    if not group_col or group_col in factors:
        group_col = next((str(col) for col in df.columns if str(col) not in factors and _is_candidate_group_column(df[col], str(col))), None)
    if group_col and not _is_candidate_group_column(df[group_col], group_col):
        group_col = next((str(col) for col in df.columns if str(col) not in factors and _is_candidate_group_column(df[col], str(col))), None)
    if not group_col:
        return None
    factor_frame = pd.DataFrame({col: _binary_numeric_series(df[col]) for col in factors})
    working = pd.concat([df[[group_col]].reset_index(drop=True), factor_frame.reset_index(drop=True)], axis=1).dropna(subset=[group_col])
    rows: list[dict[str, Any]] = []
    for group_value, part in working.groupby(group_col, dropna=False):
        active_counts = part[factors].sum(axis=1)
        prevalence = part[factors].mean().sort_values(ascending=False)
        top_factor = str(prevalence.index[0]) if len(prevalence) else ""
        rows.append({
            "group": str(group_value),
            "mean_active_factors": round(float(active_counts.mean()), 3),
            "factor_richness": int((prevalence > 0).sum()),
            "top_factor": top_factor,
            "top_factor_share": round(float(prevalence.iloc[0]), 4) if len(prevalence) else 0.0,
            "records": int(len(part)),
        })
    if len(rows) < 2:
        return None
    rows.sort(key=lambda item: (-float(item["mean_active_factors"]), str(item["group"])))
    top = rows[0]
    bottom = rows[-1]
    summary = (
        f"Factor diversity is highest for `{group_col}` = `{top['group']}` "
        f"(mean active factors {top['mean_active_factors']}, richness {top['factor_richness']}) "
        f"and lowest for `{bottom['group']}` (mean active factors {bottom['mean_active_factors']})."
    )
    chart_rows = [{"x": row["group"], "y": row["mean_active_factors"]} for row in rows]
    return _output(
        question=question,
        summary=summary,
        findings=[summary, f"Compared {len(factors)} binary factor fields across `{group_col}` groups without using prior-domain entities."],
        evidence=[f"Computed factor richness and mean active factor count from {len(factors)} binary fields."],
        limitations=["Factor diversity depends on which yes/no fields are available in the current dataset."],
        next_steps=["Review the top factor mix within the highest-diversity group."],
        code="# Compare binary factor richness across groups",
        result_preview=pd.DataFrame(rows).to_string(index=False),
        timeline=_timeline("factor_diversity_analysis", factors=len(factors), group=group_col),
        artifacts=[
            {"artifact_type": "chart", "title": f"Factor diversity by {group_col}", "content": {"chart_type": "bar", "x": "x", "y": "y", "metric": "mean active factors", "rows": chart_rows}, "visibility": "user", "pinned": True, "metadata": {"analysis_type": "factor_diversity_analysis", "grouping": group_col, "semantic_roles": {"factors": factors, "grouping": group_col}}},
            {"artifact_type": "table", "title": f"Factor diversity by {group_col}", "content": rows, "visibility": "user", "pinned": True, "metadata": {"analysis_type": "factor_diversity_analysis", "grouping": group_col}},
        ],
        trace_metadata={"fallback": "factor_diversity_analysis", "analysis_type": "factor_diversity_analysis", "operation": "FACTOR_DIVERSITY_ANALYSIS", "metric": "mean_active_factors", "dimension": group_col},
    )


def _is_candidate_group_column(series: pd.Series, name: str) -> bool:
    norm = _normalize(name)
    if _looks_identifier_like(series, name) or "id" in norm or "key" in norm or "code" in norm:
        return False
    if _binary_numeric_series(series) is not None:
        return False
    unique = int(series.nunique(dropna=True))
    if pd.api.types.is_numeric_dtype(series):
        return 3 <= unique <= 12
    return 2 <= unique <= 30


# ---------------------------------------------------------------------------
# Strategic synthesis (computed from evidence)
# ---------------------------------------------------------------------------

def _strategic_synthesis_response(
    question: str,
    df: pd.DataFrame,
) -> dict[str, Any]:
    """Compute evidence-based strategic synthesis from the dataset.

    Produces multi-signal analysis:
    - Dominant categories and concentration
    - Binary indicator prevalence scan
    - Cross-indicator/category association ranking
    - Temporal trends (if year field exists)
    """
    language = ResponseLanguagePolicy.from_message(question, protected_terms=df.columns)
    text_cols = [str(col) for col in df.columns if str(df[col].dtype) == "object"]
    numeric_cols = [str(col) for col in df.select_dtypes(include="number").columns]
    year_col = next((col for col in df.columns if _looks_year_like_column(df[col], str(col))), None)

    findings: list[str] = []
    evidence: list[str] = []

    # 1. Find the richest category column (most unique values, not an identifier)
    best_cat_col = None
    best_cat_unique = 0
    for col in text_cols:
        nunique = df[col].nunique()
        if nunique < 2 or nunique > len(df) * 0.8:  # Skip identifiers
            continue
        if _looks_identifier_like(df[col], str(col)):
            continue
        if _is_duration_like_col(df, col):
            continue
        if nunique > best_cat_unique and nunique <= 50:
            best_cat_unique = nunique
            best_cat_col = col

    # 2. Dominant categories
    if best_cat_col:
        delimiter = detect_multi_label_column(df[best_cat_col])
        if delimiter:
            exploded = explode_multi_label(df, best_cat_col, delimiter)
            counts = exploded[best_cat_col].value_counts().head(5)
        else:
            counts = df[best_cat_col].value_counts().head(5)
        top_text = ", ".join(f"`{idx}` ({cnt})" for idx, cnt in counts.items())
        total = int(counts.sum())
        top_share = float(counts.iloc[0] / max(total, 1)) if len(counts) > 0 else 0
        findings.append(
            f"Dominant `{best_cat_col}` categories: {top_text}. "
            f"The top category represents {top_share:.0%} of assignments, "
            f"{'showing concentration' if top_share > 0.3 else 'showing diversity'} in the category mix."
        )
        evidence.append(f"Counted `{best_cat_col}` across {len(df)} records.")

    # 3. Binary indicator prevalence scan
    binary_cols = _binary_indicator_columns(df, question)
    if binary_cols:
        prevalences = []
        for col in binary_cols:
            numeric = _binary_numeric_series(df[col])
            if numeric is not None:
                prevalences.append((col, float(numeric.mean()), int(numeric.sum()), int(numeric.notna().sum())))
        prevalences.sort(key=lambda x: x[1], reverse=True)
        top_indicators = prevalences[:4]
        indicator_text = ", ".join(f"`{col}` ({rate:.1%}, {pos}/{total})" for col, rate, pos, total in top_indicators)
        findings.append(
            f"Key indicator prevalence rates: {indicator_text}. "
            f"These are the most common binary conditions/flags across {len(df)} records."
        )
        evidence.append(f"Computed prevalence rates for {len(binary_cols)} binary indicator columns.")

    # 4. Cross-indicator × category association
    segment_col = best_cat_col or _find_group_col_from_question(question, df)
    if binary_cols and segment_col:
        associations: list[tuple[str, float, str, float]] = []
        for bcol in binary_cols[:6]:
            try:
                numeric = _binary_numeric_series(df[bcol])
                if numeric is None:
                    continue
                grouped = pd.DataFrame({segment_col: df[segment_col], bcol: numeric}).groupby(segment_col)[bcol].mean()
                if len(grouped) >= 2:
                    gap = float(grouped.max() - grouped.min())
                    top_group = str(grouped.idxmax())
                    associations.append((bcol, gap, top_group, float(grouped.max())))
            except Exception:
                continue
        associations.sort(key=lambda x: x[1], reverse=True)
        if associations:
            top = associations[0]
            findings.append(
                f"Strongest segment-indicator association: `{top[0]}` varies most across `{segment_col}` "
                f"(gap {top[1]:.1%}, highest in `{top[2]}` at {top[3]:.1%})."
            )
            evidence.append(f"Ranked {len(associations)} indicator × `{segment_col}` associations by gap size.")

    # 4b. Ordered demographic / ordinal gradient
    ordered_col = _ordered_group_column(question, df)
    if binary_cols and ordered_col:
        gradients: list[tuple[str, float]] = []
        groups = _ordered_groups(df[ordered_col])
        for bcol in binary_cols[:8]:
            numeric = _binary_numeric_series(df[bcol])
            if numeric is None:
                continue
            working = pd.DataFrame({"group": groups, "_value": numeric}).dropna()
            grouped = working.groupby("group")["_value"].mean().reset_index()
            if grouped["group"].nunique() < 2:
                continue
            grouped["_sort"] = grouped["group"].map(_group_sort_key)
            grouped = grouped.sort_values("_sort")
            gradients.append((bcol, float(grouped["_value"].iloc[-1] - grouped["_value"].iloc[0])))
        gradients.sort(key=lambda item: abs(item[1]), reverse=True)
        if gradients:
            direction = "increase" if gradients[0][1] >= 0 else "decrease"
            findings.append(
                f"Largest ordered-group gradient: `{gradients[0][0]}` shows a {direction} of {gradients[0][1]:.1%} "
                f"from the lowest to highest `{ordered_col}` group."
            )
            evidence.append(f"Computed ordered-group prevalence gradients across `{ordered_col}`.")

    # 4c. Data quality limitation
    missing_rates = df.isna().mean().sort_values(ascending=False)
    if len(missing_rates) and float(missing_rates.iloc[0]) > 0:
        findings.append(f"Data quality limitation: `{missing_rates.index[0]}` has the highest missingness ({float(missing_rates.iloc[0]):.1%}).")

    # 5. Temporal pattern if year field exists
    if year_col:
        year_counts = df[year_col].value_counts().sort_index()
        if len(year_counts) >= 3:
            peak_year = year_counts.idxmax()
            peak_count = int(year_counts.max())
            recent_trend = "growing" if year_counts.iloc[-1] > year_counts.iloc[-3] else "declining"
            findings.append(
                f"Peak activity in `{year_col}`={peak_year} ({peak_count} records). "
                f"Recent trend: {recent_trend}."
            )
            evidence.append(f"Analyzed record count by `{year_col}` ({len(year_counts)} unique periods).")

    # 6. Composition (type/group breakdown)
    type_col = next((col for col in text_cols if df[col].nunique() <= 5 and col != best_cat_col and not _looks_identifier_like(df[col], col)), None)
    if type_col:
        type_counts = df[type_col].value_counts()
        type_text = ", ".join(f"`{idx}` ({cnt})" for idx, cnt in type_counts.head(4).items())
        findings.append(f"Composition by `{type_col}`: {type_text}.")

    # 7. Numeric range summary (only for non-binary, non-year metrics)
    real_metrics = [col for col in numeric_cols if not _looks_year_like_column(df[col], col) and col not in binary_cols]
    if real_metrics:
        for metric in real_metrics[:1]:
            findings.append(f"`{metric}` ranges from {df[metric].min():.1f} to {df[metric].max():.1f} (mean {df[metric].mean():.1f}).")

    if any(marker in _normalize(question) for marker in ("risk", "policy", "policymaker", "executive", "important finding")):
        priority_markers = ("indicator prevalence", "prevalence rates", "association", "gradient", "missingness", "quality limitation")
        findings.sort(key=lambda item: 0 if any(marker in item.lower() for marker in priority_markers) else 1)

    # Build summary — prioritize findings with strongest signals
    summary_parts = []
    if findings:
        summary_parts.append(f"Evidence-based strategic analysis of {len(df)} records across {len(df.columns)} fields:")
        summary_parts.extend(f"{idx}. {finding}" for idx, finding in enumerate(findings[:3], 1))
    else:
        summary_parts.append(f"The dataset has {len(df)} records and {len(df.columns)} columns but lacks rich categorical or numeric fields for strategic synthesis.")

    summary = " ".join(summary_parts)
    if not findings:
        findings.append(f"The dataset has {len(df)} records. More fields or data are needed for deeper strategic analysis.")

    return _output(
        question=question,
        summary=summary,
        findings=findings[:3],
        evidence=evidence or [f"Profiled {len(df)} records across {len(df.columns)} columns."],
        limitations=["Strategic synthesis is based on available schema — additional datasets or domain context may reveal more."],
        next_steps=[
            f"Examine temporal trends in `{year_col}` for growth patterns." if year_col else "Add a temporal field to enable trend analysis.",
            f"Explore `{best_cat_col}` composition by other dimensions." if best_cat_col else "Add category fields for segmentation.",
        ],
        code="",
        result_preview="",
        timeline=_timeline("strategic_synthesis"),
        artifacts=[],
        trace_metadata={
            "fallback": "strategic_synthesis",
            "analysis_type": "strategic_synthesis",
            "dimension": best_cat_col,
            "metric": "synthesis",
        },
    )


# ---------------------------------------------------------------------------
# Find the best category and year columns from question context
# ---------------------------------------------------------------------------

def _is_duration_like_col(df: pd.DataFrame, col: str) -> bool:
    """Check if a column looks like a duration field (e.g., '90 min', '2 Seasons')."""
    col_norm = _normalize(col)
    if any(marker in col_norm for marker in ("duration", "length", "runtime", "time span")):
        return True
    if str(df[col].dtype) != "object":
        return False
    sample = df[col].dropna().astype(str).head(20)
    if sample.empty:
        return False
    duration_pattern = sample.str.contains(r'\b(?:min|season|hour|hr|sec|day)\b', case=False, regex=True, na=False)
    return float(duration_pattern.sum()) / len(sample) > 0.3


def _find_category_col_from_question(question: str, df: pd.DataFrame) -> str | None:
    """Find the best category column based on question concepts."""
    normalized = _normalize(question)
    # Check for specific concept mentions
    for concept in ("genre", "category", "type", "diagnosis", "species", "treatment", "product"):
        if concept in normalized:
            resolved = resolve_semantic_alias(concept, df)
            if resolved and not _is_duration_like_col(df, resolved):
                return resolved
    # Fall back to richest category column (excluding duration-like and identifier columns)
    text_cols = [str(col) for col in df.columns if str(df[col].dtype) == "object"]
    best_col = None
    best_score = 0
    for col in text_cols:
        nunique = df[col].nunique()
        if nunique < 2 or _looks_identifier_like(df[col], col):
            continue
        if nunique > len(df) * 0.8:
            continue
        if _is_duration_like_col(df, col):
            continue
        if nunique > best_score:
            best_score = nunique
            best_col = col
    return best_col


def _find_year_col(df: pd.DataFrame) -> str | None:
    """Find the best year-like column."""
    for col in df.columns:
        if _looks_year_like_column(df[col], str(col)):
            return str(col)
    return None


def _find_group_col_from_question(question: str, df: pd.DataFrame, exclude: str | None = None) -> str | None:
    """Find the best grouping column for diversity/comparison from question."""
    normalized = _normalize(question)
    for concept in ("region", "regional", "country", "hospital", "habitat", "area"):
        if concept in normalized:
            resolved = resolve_semantic_alias(concept, df)
            if resolved and resolved != exclude:
                return resolved
    # Fall back to a low-cardinality text column
    text_cols = [str(col) for col in df.columns if str(df[col].dtype) == "object"]
    for col in text_cols:
        if col == exclude:
            continue
        nunique = df[col].nunique()
        if 2 <= nunique <= 20 and not _looks_identifier_like(df[col], col):
            return col
    return None


def _count_based_response(
    question: str,
    df: pd.DataFrame,
    dimension_col: str,
) -> dict[str, Any]:
    """Generic count-based analysis: count records by dimension.

    Handles questions like:
    - releases by year → count records by year
    - genre dominance → count by genre/category
    - country contribution → count by country
    - movie vs TV show distribution → count by type
    """
    language = ResponseLanguagePolicy.from_message(question, protected_terms=df.columns)
    # Detect multi-label columns and split them into individual labels.
    multi_label_note = ""
    delimiter = detect_multi_label_column(df[dimension_col]) if dimension_col in df.columns else None
    if delimiter:
        exploded = explode_multi_label(df, dimension_col, delimiter)
        multi_label_note = " (multi-label values split into individual labels)"
        grouped = (
            exploded.groupby(dimension_col, dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
    else:
        grouped = (
            df.groupby(dimension_col, dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
    total = int(grouped["count"].sum())
    grouped["share"] = grouped["count"] / max(total, 1)
    top = grouped.iloc[0] if not grouped.empty else None
    bottom = grouped.iloc[-1] if not grouped.empty else None
    top_groups = ", ".join(
        f"`{row[dimension_col]}` ({int(row['count'])})"
        for _, row in grouped.head(3).iterrows()
    )
    bottom_groups = ", ".join(
        f"`{row[dimension_col]}` ({int(row['count'])})"
        for _, row in grouped.tail(3).sort_values("count").iterrows()
    )
    summary = (
        f"Record count by `{dimension_col}`{multi_label_note}: the most common values are {top_groups}. "
        f"The least common are {bottom_groups}. "
        f"There are {len(grouped)} unique values across {total:,} total records."
    )
    result_rows = grouped.head(25).to_dict(orient="records")
    chart_rows = grouped.head(20).to_dict(orient="records")
    timeline = _timeline("count_based_analysis", dimension_col=dimension_col, rows=total, groups=len(grouped))
    top_share = f"The top `{dimension_col}` contributes {float(top['share']):.1%} of all records." if top is not None else ""
    return _output(
        question=question,
        summary=summary,
        findings=[
            summary,
            top_share,
        ],
        evidence=[f"Counted records by `{dimension_col}` over {total:,} rows."],
        limitations=["This counts records, not a specific numeric metric. Use a numeric column if aggregation (sum, average) is needed."],
        next_steps=[
            f"Break down the `{dimension_col}` count by another dimension for deeper insight.",
            "Compare counts with a metric (if available) to see if high-count categories also have high values.",
        ],
        code=f"result = df.groupby({dimension_col!r}, dropna=False).size().reset_index(name='count').sort_values('count', ascending=False)",
        result_preview=pd.DataFrame(result_rows).to_string(index=False),
        timeline=timeline,
        artifacts=[
            {
                "artifact_type": "chart",
                "title": f"Record count by {dimension_col}",
                "content": {
                    "chart_type": "bar",
                    "x": dimension_col,
                    "y": "count",
                    "metric": "record_count",
                    "dimension": dimension_col,
                    "aggregation": "count",
                    "rows": chart_rows,
                    "row_count": total,
                },
                "visibility": "user",
                "pinned": True,
                "metadata": {
                    "metric": "record_count",
                    "dimension": dimension_col,
                    "aggregation": "count",
                    "analysis_type": "count_based",
                },
            },
            {
                "artifact_type": "table",
                "title": f"Records by {dimension_col}",
                "content": result_rows,
                "visibility": "user",
                "metadata": {"metric": "record_count", "dimension": dimension_col},
            },
        ],
        trace_metadata={
            "fallback": "count_based_analysis",
            "analysis_type": "count_based",
            "dimension": dimension_col,
            "metric": "record_count",
        },
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
    result_preview = grouped.to_string(index=False)
    top_chart = grouped.head(20)
    if language.is_russian:
        # English-only output policy — use same wording as the English branch
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
            f"`{widest[dimension_col]}` has the widest internal spread "
            f"({widest['min']:.2f} to {widest['max']:.2f}), so check whether that comes from a few extreme values or a genuinely mixed subgroup. "
            f"Validate this gap with a record-count or outlier check before using it in a decision. {reliability_note}"
        )
        evidence = [f"Grouped `{metric_col}` by `{dimension_col}` over {len(df)} rows."]
        limitations = [
            "This is descriptive aggregation over the current dataset; it does not prove causality.",
            reliability_note,
            f"`{widest[dimension_col]}` needs validation because a wide spread can come from a few extreme values or a genuinely mixed subgroup.",
        ]
        next_steps = [
            f"Check whether `{widest[dimension_col]}` remains unusually variable after removing extreme rows.",
            f"Break `{metric_col}` by another relevant field and check whether the gap across `{dimension_col}` persists.",
            f"Use a grouped bar chart to support reporting on `{dimension_col}` differences.",
        ]
        chart_title = average_chart_title(metric_col, dimension_col, language) if aggregation == "mean" else f"{metric_col} by {dimension_col}"
        ranking_scope = f"Top {min(len(top_chart), 20)} groups of `{dimension_col}` by {value_label} `{metric_col}` out of {len(grouped)} groups."
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
            f"`{widest[dimension_col]}` has the widest internal spread "
            f"({widest['min']:.2f} to {widest['max']:.2f}), so check whether that comes from a few extreme values or a genuinely mixed subgroup. "
            f"Validate this gap with a record-count or outlier check before using it in a decision. {reliability_note}"
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
            f"`{widest[dimension_col]}` needs validation because a wide spread can come from a few extreme values or a genuinely mixed subgroup.",
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


def _schema_profile_heatmap(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    timestamp_cols: list[str],
    *,
    title: str = "Schema Profile",
) -> dict[str, Any]:
    """Generate a heatmap chart artifact showing field-type distribution."""
    field_names = list(df.columns)[:15]
    num_set = set(numeric_cols)
    cat_set = set(categorical_cols)
    ts_set = set(timestamp_cols)
    rows: list[dict[str, Any]] = []
    for field in field_names:
        rows.append({"row": str(field), "column": "Numeric", "value": 1.0 if field in num_set else 0.0})
        rows.append({"row": str(field), "column": "Categorical", "value": 1.0 if field in cat_set else 0.0})
        rows.append({"row": str(field), "column": "Temporal", "value": 1.0 if field in ts_set else 0.0})
    return {
        "artifact_type": "chart",
        "title": title,
        "visibility": "user",
        "content": {
            "chart_type": "heatmap",
            "title": title,
            "x": "column",
            "y": "row",
            "rows": rows,
        },
    }


def _field_classification_table(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    timestamp_cols: list[str],
) -> dict[str, Any]:
    """Generate a table artifact showing field classification (fact/dimension/temporal)."""
    rows: list[dict[str, str]] = []
    for col in list(df.columns)[:20]:
        col_str = str(col)
        if col_str in set(numeric_cols):
            role = "Fact (numeric)"
        elif col_str in set(timestamp_cols):
            role = "Temporal"
        elif col_str in set(categorical_cols):
            role = "Dimension"
        else:
            role = "Other"
        rows.append({"Field": col_str, "Role": role, "Distinct": str(df[col].nunique())})
    return {
        "artifact_type": "table",
        "title": "Field Classification",
        "visibility": "user",
        "content": rows,
    }


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
        f"Typical group size is {median_count:.0f} rows, so the larger groups can support more stable comparisons."
    )


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
    context_policy = str(state.get("context_policy") or (conversation_context.get("routing_decision") or {}).get("context_policy") or "")
    allow_active_state = context_policy in {"", "continue"}
    active_metric = (_valid_column(df, state.get("active_metric")) or _valid_column(df, chart.get("metric"))) if allow_active_state else None
    active_dimension = (_valid_column(df, state.get("active_dimension")) or _valid_column(df, chart.get("dimension"))) if allow_active_state else None
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
        if allow_active_state:
            return explicit_metric or active_metric, explicit_dimension or active_dimension, "explicit_topic_switch"
        return explicit_metric, explicit_dimension, "explicit_topic_switch"
    if explicit_metric or explicit_dimension:
        if allow_active_state:
            return explicit_metric or active_metric, explicit_dimension or active_dimension, "explicit_user_instruction"
        return explicit_metric, explicit_dimension, "explicit_user_instruction"
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


def _stem_match(a: str, b: str) -> bool:
    """Check if two words share the same root/stem (simple heuristic).

    Handles common English patterns:
    - diagnosis ↔ diagnoses
    - symptom ↔ symptoms
    - category ↔ categories
    - treatment ↔ treatments
    """
    if a == b:
        return True
    if not a or not b:
        return False
    # Simple prefix match (e.g., treatment → treatments)
    if a.startswith(b) or b.startswith(a):
        return True
    # Shared stem of at least 4 characters
    min_len = min(len(a), len(b))
    if min_len < 4:
        return False
    shared = 0
    for i in range(min_len):
        if a[i] == b[i]:
            shared += 1
        else:
            break
    return shared >= min(4, min_len - 1)


def _count_based_dimension(
    question: str,
    df: pd.DataFrame,
    dimension_col: str | None,
    timestamp_col: str | None,
) -> str | None:
    """Choose the best grouping dimension for count-based analysis.

    When the user says 'releases by year', the dimension should be the year column,
    not a random categorical dimension like 'type'. This function checks if the question
    mentions a year/time concept and uses the matching column.
    """
    normalized = _normalize(question)
    # Strip punctuation from tokens for clean matching
    question_tokens = {token.strip("?.,!;:()\"'") for token in normalized.split()}
    # Check if the question mentions year/time — prefer year-like or timestamp columns
    year_markers = ("year", "years", "по годам", "годам", "год")
    if any(marker in normalized for marker in year_markers):
        # Try to find a year-like numeric column
        for col in df.columns:
            col_norm = _normalize(str(col))
            if "year" in col_norm or "год" in col_norm:
                return str(col)
        # Fall back to timestamp column
        if timestamp_col:
            return timestamp_col
    # Check if question mentions a specific column by name (exact substring)
    for col in df.columns:
        col_norm = _normalize(str(col))
        if col_norm and col_norm in normalized:
            return str(col)
    # Token-based matching with basic plural/stem handling
    # 'diagnosis' matches 'diagnoses', 'symptom' matches 'symptoms', etc.
    best_col: str | None = None
    best_score = 0
    text_and_dim_columns = [
        str(col) for col in df.columns
        if str(df[col].dtype) == "object" or str(col) == (dimension_col or "")
    ]
    for col in text_and_dim_columns:
        col_norm = _normalize(col)
        col_tokens = set(col_norm.replace("_", " ").split())
        if not col_tokens:
            continue
        score = 0
        for token in col_tokens:
            if token in question_tokens:
                score += 3
            # Stem matching: diagnosis~diagnoses, symptom~symptoms, treatment~treatments
            elif any(_stem_match(token, qt) for qt in question_tokens if len(qt) >= 3):
                score += 2
        if score > best_score:
            best_score = score
            best_col = col
    if best_col and best_score >= 2:
        return best_col
    # Fall back to dimension_col
    return dimension_col


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
