from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from source.product.conversation import (
    build_conversation_state,
    is_semantically_redundant_response as conversation_redundant_response,
    resolve_user_intent,
)
from source.product.conversation_engine import answer_from_conversation_state, clarification_from_state, response_quality_gate
from source.product.data_context import build_data_source_usage_context, usage_context_to_prompt
from source.product.dataset_registry import (
    DatasetResolutionResult,
    DatasetScope,
    build_investigation_dataset_registry,
    clarification_output,
    cross_dataset_output,
    registry_prompt,
    resolve_dataset_scope,
    resolve_explicit_metric_dataset,
)
from source.product.execution_context import (
    ExecutionContextUnavailableError,
    execution_context_failure_output,
    execution_required_for_question,
    resolve_dataset_runtime,
)
from source.product.analytical_graph import synthesize_transformation_change, transformation_from_payload
from source.product.branch_workspace import BranchWorkspaceManager, activate_branch, upsert_branch_for_intent, upsert_branch_from_plan
from source.product.evidence_resolution import (
    BranchIdentity,
    build_active_target,
    evidence_response_text,
    is_evidence_followup,
    resolve_evidence_subject,
    transformation_state_from_payload,
)
from source.product.fallback_analysis import deterministic_investigation_fallback
from source.product.language_policy import ResponseLanguagePolicy
from source.product.llm_reasoning import sanitize_user_visible_text
from source.product.non_analytical import non_analytical_output, sanitize_non_analytical_text
from source.product.question_routing import ContextPolicy, decide_routing, should_ignore_active_branch
from source.product.investigation import (
    InvestigationMessage,
    InvestigationMessageRole,
    InvestigationMessageType,
    InvestigationRun,
    InvestigationRunEvent,
    InvestigationRunEventSeverity,
    InvestigationRunEventType,
    InvestigationRunStage,
    InvestigationRunStatus,
    InvestigationStatus,
    utc_now,
)
from source.product.run_validation import validate_investigation_result
from source.product.semantic_layer import build_investigation_thread_state
from source.product.service import InvestigationService
from source.product.store import InvestigationStore


class InvestigationRunService:
    def __init__(
        self,
        store: InvestigationStore | None = None,
        investigation_service: InvestigationService | None = None,
    ) -> None:
        self.store = store or InvestigationStore()
        self.investigation_service = investigation_service or InvestigationService(self.store)

    def run_investigation(
        self,
        investigation_id: str,
        data_source_ids: list[str] | None = None,
        force_refresh_context: bool = False,
        df: Any = None,
        message_id: str | None = None,
        analysis_mode: str = "exploration",
    ) -> InvestigationRun:
        investigation = self.store.get_investigation(investigation_id)
        active_message = self.store.get_investigation_message(message_id) if message_id else None
        if active_message and active_message.investigation_id != investigation_id:
            raise KeyError(f"InvestigationMessage not found for investigation: {message_id}")
        resolved_source_ids = _unique_ids(list(investigation.linked_data_source_ids) + list(data_source_ids or []))
        run = InvestigationRun(
            investigation_id=investigation_id,
            status=InvestigationRunStatus.QUEUED,
            current_stage=InvestigationRunStage.PREPARING_DATA,
            data_source_ids=resolved_source_ids,
            started_at=None,
            metadata={
                "force_refresh_context": force_refresh_context,
                "message_id": message_id,
                "analysis_mode": _normalize_analysis_mode(analysis_mode),
            },
        )
        self.store.create_investigation_run(run)

        try:
            self._transition(run, InvestigationRunStatus.RUNNING, InvestigationRunStage.PREPARING_DATA)
            for data_source_id in resolved_source_ids:
                self.store.get_data_source(data_source_id)
                self.store.link_data_source_to_investigation(investigation_id, data_source_id)
            self._emit(
                run,
                InvestigationRunEventType.INFO,
                "Prepared data sources for this run.",
                metadata={"data_source_ids": resolved_source_ids},
            )

            self._transition(run, InvestigationRunStatus.RUNNING, InvestigationRunStage.BUILDING_CONTEXT)
            usage_contexts = [build_data_source_usage_context(self.store, item) for item in resolved_source_ids]
            _activate_branch_from_message_metadata(self.store, investigation_id, active_message)
            conversation_context = _build_conversation_context(
                self.store,
                investigation_id,
                active_message_id=message_id,
                active_question=active_message.content if active_message else investigation.user_question,
            )
            active_question = active_message.content if active_message else investigation.user_question
            routing_decision = decide_routing(
                active_question,
                has_active_context=bool((conversation_context.get("conversation_state") or {}).get("active_branch_id") if isinstance(conversation_context, dict) else False),
            )
            conversation_context["routing_decision"] = routing_decision.to_payload()
            if isinstance(conversation_context.get("conversation_state"), dict):
                conversation_context["conversation_state"]["question_intent_type"] = routing_decision.question_intent_type.value
                conversation_context["conversation_state"]["branch_action"] = routing_decision.branch_action.value
            dataset_registry = build_investigation_dataset_registry(self.store, investigation, resolved_source_ids)
            if not dataset_registry:
                dataset_resolution = DatasetResolutionResult(
                    DatasetScope.SINGLE,
                    [],
                    1.0,
                    ["No product data source registry is attached; using legacy run context."],
                )
            else:
                dataset_resolution = resolve_dataset_scope(
                    question=active_question,
                    registry=dataset_registry,
                    conversation_context=conversation_context,
                    user_selected_dataset_ids=_message_selected_dataset_ids(active_message),
                )
            run.metadata["dataset_resolution"] = dataset_resolution.to_dict()
            run.metadata["dataset_registry"] = [entry.to_dict() for entry in dataset_registry]
            run.run_context_summary = _build_run_context_summary(usage_contexts, conversation_context)
            run.run_context_summary["dataset_resolution"] = dataset_resolution.to_dict()
            self.store.update_investigation_run(run)
            self._emit(
                run,
                InvestigationRunEventType.INFO,
                f"Built usage context for {len(usage_contexts)} data source(s).",
                metadata={"data_sources_count": len(usage_contexts)},
            )
            for context in usage_contexts:
                if not context.sample_rows:
                    self._emit(
                        run,
                        InvestigationRunEventType.WARNING,
                        f"{context.name}: no sample rows are available.",
                        severity=InvestigationRunEventSeverity.WARNING,
                        metadata={"data_source_id": context.data_source_id},
                    )
                if not any(
                    "Business meaning:" in note or "Display name:" in note
                    for column in context.column_summaries
                    for note in column.notes
                ):
                    self._emit(
                        run,
                        InvestigationRunEventType.WARNING,
                        f"{context.name}: no semantic column notes were found.",
                        severity=InvestigationRunEventSeverity.WARNING,
                        metadata={"data_source_id": context.data_source_id},
                    )
                for caveat in context.caveats[:8]:
                    self._emit(
                        run,
                        InvestigationRunEventType.WARNING,
                        f"{context.name}: {caveat}",
                        severity=InvestigationRunEventSeverity.WARNING,
                        metadata={"data_source_id": context.data_source_id},
                    )

            self._transition(run, InvestigationRunStatus.RUNNING, InvestigationRunStage.RUNNING_ANALYSIS)
            analysis_df = df
            execution_contexts: list[dict[str, Any]] = []
            precomputed_output: dict[str, Any] | None = None
            runtime_frames: dict[str, Any] = {}
            if routing_decision.execution_mode == "conversational":
                precomputed_output = non_analytical_output(
                    active_question,
                    routing_decision.question_intent_type,
                    has_dataset_context=bool(resolved_source_ids),
                )
                self._emit(
                    run,
                    InvestigationRunEventType.INFO,
                    "Handled as a non-analytical conversational request.",
                    metadata=routing_decision.to_payload(),
                )
            elif dataset_resolution.scope == DatasetScope.AMBIGUOUS:
                # --- GLOBAL EXPLICIT METRIC SEARCH ---
                # Before asking for clarification, check if the user explicitly
                # mentioned a metric that uniquely exists in one dataset.
                # If so, override the ambiguous result and use that dataset.
                explicit_match = resolve_explicit_metric_dataset(active_question, dataset_registry)
                if explicit_match:
                    override_id, override_column = explicit_match
                    dataset_resolution = DatasetResolutionResult(
                        DatasetScope.SINGLE,
                        [override_id],
                        0.88,
                        [f"Explicit metric `{override_column}` uniquely found in one dataset."],
                        candidate_scores=dataset_resolution.candidate_scores,
                    )
                    run.metadata["dataset_resolution"] = dataset_resolution.to_dict()
                    self._emit(
                        run,
                        InvestigationRunEventType.INFO,
                        f"Resolved ambiguous scope via explicit metric `{override_column}`.",
                        metadata={"override_dataset_id": override_id, "override_column": override_column},
                    )
                else:
                    precomputed_output = clarification_output(active_question, dataset_resolution)
                    self._emit(
                        run,
                        InvestigationRunEventType.WARNING,
                        "Dataset choice is ambiguous; asking for clarification.",
                        severity=InvestigationRunEventSeverity.WARNING,
                        metadata=dataset_resolution.to_dict(),
                    )
            elif analysis_df is None:
                if dataset_resolution.scope == DatasetScope.CROSS:
                    runtime_frames, execution_contexts = self._load_dataframes_for_run(dataset_resolution.selected_dataset_ids, run)
                    missing = [item for item in dataset_resolution.selected_dataset_ids if item not in runtime_frames]
                    if missing:
                        run.metadata["execution_context_failures"] = list(run.metadata.get("execution_context_failures") or [])
                        first_failure = (run.metadata.get("execution_context_failures") or [{}])[0]
                        precomputed_output = execution_context_failure_output(
                            active_question,
                            ExecutionContextUnavailableError(
                                str(first_failure.get("message") or "Executable dataset rows are unavailable for one selected dataset."),
                                data_source_id=str(first_failure.get("data_source_id") or missing[0]),
                                reason=str(first_failure.get("reason") or "raw_rows_unavailable"),
                            ),
                        )
                    else:
                        prior_investigation = self.store.get_investigation(investigation_id)
                        prior_finding_texts = [f.text for f in prior_investigation.findings[-20:]]
                        precomputed_output = cross_dataset_output(
                            question=active_question,
                            registry=dataset_registry,
                            frames=runtime_frames,
                            result=dataset_resolution,
                            prior_findings=prior_finding_texts,
                        )
                        trace = precomputed_output.get("trace_metadata") if isinstance(precomputed_output.get("trace_metadata"), dict) else {}
                        if isinstance(trace.get("dataset_relationships"), dict):
                            run.metadata["dataset_relationships"] = trace["dataset_relationships"]
                            run.run_context_summary["dataset_relationships"] = trace["dataset_relationships"]
                else:
                    analysis_df, execution_contexts = self._load_dataframe_for_run(dataset_resolution.selected_dataset_ids, run)
            # --- EXPLICIT METRIC DATASET OVERRIDE ---
            # When a df was passed by the caller but the canonical resolver
            # selected a specific dataset (explicit metric match), check
            # whether the passed df actually contains the expected metric.
            # If not, attempt to load the correct df from runtime store.
            if analysis_df is not None and dataset_resolution.scope == DatasetScope.SINGLE and dataset_registry:
                explicit_match = resolve_explicit_metric_dataset(active_question, dataset_registry)
                if explicit_match:
                    _override_id, override_column = explicit_match
                    if hasattr(analysis_df, "columns") and override_column not in set(str(c) for c in analysis_df.columns):
                        override_df, _ctx = self._load_dataframe_for_run([_override_id], run)
                        if override_df is not None:
                            analysis_df = override_df
                            dataset_resolution = DatasetResolutionResult(
                                DatasetScope.SINGLE,
                                [_override_id],
                                0.90,
                                [f"Overrode caller-provided df: explicit metric `{override_column}` found in a different loaded dataset."],
                                candidate_scores=dataset_resolution.candidate_scores,
                            )
                            run.metadata["dataset_resolution"] = dataset_resolution.to_dict()
                            self._emit(
                                run,
                                InvestigationRunEventType.INFO,
                                f"Switched to dataset containing explicit metric `{override_column}`.",
                                metadata={"override_dataset_id": _override_id, "override_column": override_column},
                            )
            updated = self.investigation_service.run_investigation(
                investigation_id,
                df=analysis_df,
                data_context={
                    "data_source_ids": dataset_resolution.selected_dataset_ids or resolved_source_ids,
                    "attached_data_source_ids": resolved_source_ids,
                    "product_run_id": run.run_id,
                    "active_question": active_question,
                    "active_message_id": message_id,
                    "analysis_mode": _normalize_analysis_mode(analysis_mode),
                    "conversation_context": conversation_context,
                    "data_source_usage_contexts": [_jsonable(context) for context in usage_contexts],
                    "data_context_prompt": usage_context_to_prompt(usage_contexts) + "\n\n" + registry_prompt(dataset_registry),
                    "execution_contexts": execution_contexts,
                    "execution_context_failures": list(run.metadata.get("execution_context_failures") or []),
                    "dataset_registry": [entry.to_dict() for entry in dataset_registry],
                    "dataset_resolution": dataset_resolution.to_dict(),
                    "precomputed_output": precomputed_output,
                },
            )

            self._transition(run, InvestigationRunStatus.RUNNING, InvestigationRunStage.VALIDATING_RESULTS)
            latest_output = updated.runs[-1].output if getattr(updated, "runs", None) else {}
            latest_trace = latest_output.get("trace_metadata") if isinstance(latest_output, dict) and isinstance(latest_output.get("trace_metadata"), dict) else {}
            if latest_trace.get("dataset_execution_scopes"):
                run.metadata["dataset_execution_scopes"] = latest_trace["dataset_execution_scopes"]
                run.metadata["multi_dataset_lifecycle_trace"] = _multi_dataset_lifecycle_trace(
                    question=active_question,
                    dataset_registry=dataset_registry,
                    dataset_resolution=dataset_resolution,
                    routing_decision=routing_decision,
                    trace_metadata=latest_trace,
                    artifacts=getattr(updated, "artifacts", []) or [],
                    findings=getattr(updated, "findings", []) or [],
                    run_id=run.run_id,
                )
            validation = validate_investigation_result(updated, run_id=run.run_id)
            run.run_context_summary["validation"] = validation
            run.artifact_ids = [artifact.artifact_id for artifact in updated.artifacts if artifact.run_id == run.run_id]
            for artifact_id in run.artifact_ids:
                self._emit(
                    run,
                    InvestigationRunEventType.ARTIFACT_CREATED,
                    "Created an investigation artifact.",
                    metadata={"artifact_id": artifact_id},
                )
            for key in ["findings_count", "artifacts_count", "reports_generated", "execution_error_present"]:
                severity = (
                    InvestigationRunEventSeverity.ERROR
                    if key == "execution_error_present" and validation.get(key)
                    else InvestigationRunEventSeverity.INFO
                )
                self._emit(
                    run,
                    InvestigationRunEventType.VALIDATION_CHECK,
                    f"Validation check: {key} = {validation.get(key)}.",
                    severity=severity,
                    metadata={"check": key, "value": validation.get(key)},
                )
            for warning in validation.get("warnings", []):
                self._emit(
                    run,
                    InvestigationRunEventType.VALIDATION_CHECK,
                    warning,
                    severity=InvestigationRunEventSeverity.WARNING,
                    metadata={"kind": "validation_warning"},
                )

            self._transition(run, InvestigationRunStatus.RUNNING, InvestigationRunStage.GENERATING_REPORT)
            if updated.report and updated.report.run_id == run.run_id:
                run.report_ids = [updated.report.report_id]
                self._emit(
                    run,
                    InvestigationRunEventType.REPORT_CREATED,
                    "Generated a DecisionReport.",
                    metadata={"report_id": updated.report.report_id},
                )
            if updated.status == InvestigationStatus.FAILED:
                latest_error = updated.runs[-1].error if updated.runs else None
                raise RuntimeError(latest_error or "Investigation run failed.")

            self._emit_stage_completed(run, run.current_stage)
            run.status = InvestigationRunStatus.COMPLETED
            run.current_stage = InvestigationRunStage.COMPLETED
            run.completed_at = utc_now()
            run.finished_at = run.completed_at
            self._emit(
                run,
                InvestigationRunEventType.STAGE_COMPLETED,
                "Latest analytical pass is available.",
                stage=InvestigationRunStage.COMPLETED,
            )
            final_run = self.store.update_investigation_run(run)
            _persist_conversation_state(
                self.store,
                investigation_id,
                question=active_question,
                run_id=run.run_id,
            )
            self.ensure_assistant_response_for_user_message(
                investigation_id,
                final_run.run_id,
                updated,
                active_message_id=message_id,
                question=active_question,
                df=analysis_df,
            )
            return final_run
        except Exception as exc:
            run.status = InvestigationRunStatus.FAILED
            run.error_message = f"{type(exc).__name__}: {exc}"
            run.error = run.error_message
            run.completed_at = utc_now()
            run.finished_at = run.completed_at
            self._emit(
                run,
                InvestigationRunEventType.ERROR,
                "Investigation run failed.",
                severity=InvestigationRunEventSeverity.ERROR,
                metadata={"error": run.error_message},
            )
            self.store.update_investigation_run(run)
            self.ensure_assistant_response_for_user_message(
                investigation_id,
                run.run_id,
                None,
                active_message_id=message_id,
                question=active_message.content if active_message else investigation.user_question,
                error_message=run.error_message,
            )
            return self.store.get_investigation_run(run.run_id)

    def _load_dataframe_for_run(self, data_source_ids: list[str], run: InvestigationRun) -> tuple[Any, list[dict[str, Any]]]:
        frames, contexts = self._load_dataframes_for_run(data_source_ids, run, stop_after_first=True)
        if frames:
            first_id = next(iter(frames))
            return frames[first_id], contexts
        return None, contexts

    def _load_dataframes_for_run(
        self,
        data_source_ids: list[str],
        run: InvestigationRun,
        *,
        stop_after_first: bool = False,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        failures: list[dict[str, Any]] = []
        frames: dict[str, Any] = {}
        execution_contexts: list[dict[str, Any]] = []
        for data_source_id in data_source_ids:
            try:
                source = self.store.get_data_source(data_source_id)
            except KeyError:
                continue
            try:
                df, execution_context = resolve_dataset_runtime(self.store, data_source_id)
            except ExecutionContextUnavailableError as exc:
                failures.append(exc.to_payload())
                self._emit(
                    run,
                    InvestigationRunEventType.WARNING,
                    f"{source.name}: executable dataset context is unavailable.",
                    severity=InvestigationRunEventSeverity.WARNING,
                    metadata=exc.to_payload(),
                )
                continue
            self._emit(
                run,
                InvestigationRunEventType.INFO,
                f"Resolved executable dataset context for {source.name}.",
                metadata={
                    "data_source_id": data_source_id,
                    "runtime_reference": execution_context.dataset_runtime_reference,
                    "storage_reference": execution_context.storage_reference,
                    "rows": int(len(df)),
                    "columns": int(len(getattr(df, "columns", []))),
                },
            )
            frames[data_source_id] = df
            execution_contexts.append(execution_context.to_dict())
            if stop_after_first:
                return frames, execution_contexts
        if failures:
            run.metadata["execution_context_failures"] = failures
        return frames, execution_contexts

    def _transition(
        self,
        run: InvestigationRun,
        status: InvestigationRunStatus,
        stage: InvestigationRunStage,
    ) -> InvestigationRun:
        previous_stage = run.current_stage
        if run.started_at is not None and previous_stage != stage:
            self._emit_stage_completed(run, previous_stage)
        run.status = status
        run.current_stage = stage
        if run.started_at is None:
            run.started_at = utc_now()
        updated = self.store.update_investigation_run(run)
        self._emit(
            run,
            InvestigationRunEventType.STAGE_STARTED,
            _stage_label(stage) + " started.",
            stage=stage,
        )
        return updated

    def _emit_stage_completed(self, run: InvestigationRun, stage: InvestigationRunStage) -> None:
        self._emit(
            run,
            InvestigationRunEventType.STAGE_COMPLETED,
            _stage_label(stage) + " updated.",
            stage=stage,
        )

    def _emit(
        self,
        run: InvestigationRun,
        event_type: InvestigationRunEventType,
        message: str,
        stage: InvestigationRunStage | None = None,
        severity: InvestigationRunEventSeverity = InvestigationRunEventSeverity.INFO,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        try:
            self.store.add_investigation_run_event(
                InvestigationRunEvent(
                    run_id=run.run_id,
                    investigation_id=run.investigation_id,
                    event_type=event_type,
                    stage=stage or run.current_stage,
                    message=message,
                    severity=severity,
                    metadata=metadata or {},
                )
            )
        except Exception:
            return

    def ensure_assistant_response_for_user_message(
        self,
        investigation_id: str,
        run_id: str,
        investigation: Any | None,
        *,
        active_message_id: str | None = None,
        question: str = "",
        error_message: str | None = None,
        df: Any = None,
    ) -> None:
        try:
            try:
                investigation = self.store.get_investigation(investigation_id)
            except Exception:
                pass
            messages = self.store.list_investigation_messages(investigation_id)
            for message in messages:
                if message.role != InvestigationMessageRole.ASSISTANT:
                    continue
                if active_message_id and message.metadata.get("response_to_message_id") == active_message_id:
                    return
                if not active_message_id and message.run_id == run_id:
                    return
            if error_message:
                self._add_error_message(investigation_id, run_id, error_message, active_message_id=active_message_id)
                return
            summary = _summarize_run_result(investigation)
            execution_context_error_summary = _is_execution_context_unavailable_summary(summary, investigation)
            # ── Terminal Decision Guard ──
            # If the latest run produced a terminal decision (e.g. semantic_incompatibility),
            # skip ALL downstream processing. The terminal message is the final answer.
            _is_terminal = bool(_extract_terminal_decision_from_runs(investigation))
            if not _is_terminal:
                if df is not None and _looks_like_state_clarification(summary):
                    conversation_context = {}
                    if investigation is not None and isinstance(getattr(investigation, "metadata", None), dict):
                        conversation_context = {
                            "conversation_state": investigation.metadata.get("conversation_state") or {},
                            "recent_artifacts": list(getattr(investigation, "artifacts", []) or []),
                        }
                    deterministic = deterministic_investigation_fallback(
                        question,
                        df,
                        data_context={"conversation_context": conversation_context},
                    )
                    deterministic_summary = _output_summary(deterministic)
                    if deterministic_summary:
                        summary = deterministic_summary
                previous_assistant = [
                    message
                    for message in messages
                    if message.role == InvestigationMessageRole.ASSISTANT
                    and not (active_message_id and message.metadata.get("response_to_message_id") == active_message_id)
                ]
                state_payload = {}
                if investigation is not None and isinstance(getattr(investigation, "metadata", None), dict):
                    state_payload = investigation.metadata.get("conversation_state") or {}
                if (
                    not execution_context_error_summary
                    and (not _is_chart_request(question))
                ) and is_semantically_redundant_response(
                    summary,
                    [message.content for message in previous_assistant[-8:]],
                    state_payload,
                ):
                    if not _is_self_sufficient_dataset_scan(question, summary):
                        summary = _non_redundant_follow_up_response(question, investigation, summary, df=df)
                        if is_semantically_redundant_response(
                            summary,
                            [message.content for message in previous_assistant[-8:]],
                            state_payload,
                        ):
                            summary = _duplicate_follow_up_deepening(question, investigation, df=df) or summary
                conversation_context = {}
                if investigation is not None and isinstance(getattr(investigation, "metadata", None), dict):
                    conversation_context = {
                        "conversation_state": state_payload,
                        "recent_artifacts": list(getattr(investigation, "artifacts", []) or []),
                    }
                valid, _reason = response_quality_gate(
                    question=question,
                    response_text=summary,
                    conversation_context=conversation_context,
                )
                if not valid and not execution_context_error_summary:
                    deterministic_summary = ""
                    if df is not None:
                        deterministic = deterministic_investigation_fallback(
                            question,
                            df,
                            data_context={"conversation_context": conversation_context},
                        )
                        deterministic_summary = _output_summary(deterministic)
                    if deterministic_summary:
                        summary = deterministic_summary
                    else:
                        engine_response = answer_from_conversation_state(
                            question=question,
                            conversation_context=conversation_context,
                            recent_artifacts=list(getattr(investigation, "artifacts", []) or []) if investigation is not None else [],
                        )
                        if engine_response:
                            summary = engine_response.text
                        else:
                            clarification = clarification_from_state(conversation_context)
                            if clarification:
                                summary = clarification.text
                if df is not None and _looks_like_state_clarification(summary) and not execution_context_error_summary:
                    deterministic = deterministic_investigation_fallback(
                        question,
                        df,
                        data_context={"conversation_context": conversation_context},
                    )
                    deterministic_summary = _output_summary(deterministic)
                    if deterministic_summary:
                        summary = deterministic_summary
            summary = sanitize_user_visible_text(summary)
            routing_payload = (getattr(investigation, "metadata", {}) or {}).get("conversation_state") if investigation is not None else {}
            question_intent = str((routing_payload or {}).get("question_intent_type") or "")
            if question_intent:
                summary = sanitize_non_analytical_text(summary, question, question_intent)
            self._add_run_summary_message(
                investigation_id,
                run_id,
                investigation,
                active_message_id=active_message_id,
                question=question,
                summary=summary,
            )
        except Exception:
            try:
                self._add_error_message(
                    investigation_id,
                    run_id,
                    "A visible assistant response could not be generated for this question.",
                    active_message_id=active_message_id,
                )
            except Exception:
                return

    def _add_run_summary_message(
        self,
        investigation_id: str,
        run_id: str,
        investigation: Any,
        *,
        active_message_id: str | None = None,
        question: str = "",
        summary: str | None = None,
    ) -> None:
        try:
            summary = summary or _summarize_run_result(investigation)
            self.store.add_investigation_message(
                InvestigationMessage(
                    investigation_id=investigation_id,
                    run_id=run_id,
                    role=InvestigationMessageRole.ASSISTANT,
                    message_type=InvestigationMessageType.RUN_SUMMARY,
                    content=summary,
                    metadata={
                        "kind": "run_summary",
                        "response_to_message_id": active_message_id,
                        "question": question,
                    },
                )
            )
        except Exception:
            return

    def _add_error_message(
        self,
        investigation_id: str,
        run_id: str,
        error_message: str | None,
        *,
        active_message_id: str | None = None,
    ) -> None:
        try:
            self.store.add_investigation_message(
                InvestigationMessage(
                    investigation_id=investigation_id,
                    run_id=run_id,
                    role=InvestigationMessageRole.ASSISTANT,
                    message_type=InvestigationMessageType.ERROR,
                    content=error_message or "Investigation run failed.",
                    metadata={"kind": "run_error", "response_to_message_id": active_message_id},
                )
            )
        except Exception:
            return


def _build_run_context_summary(contexts: list[Any], conversation_context: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "data_sources_count": len(contexts),
        "data_sources": [
            {
                "data_source_id": context.data_source_id,
                "name": context.name,
                "status": context.status.value,
                "schema_summary": context.schema_summary,
                "caveats": list(context.caveats),
                "previous_questions": list(context.previous_questions),
                "semantic_column_notes_count": sum(
                    1
                    for column in context.column_summaries
                    if any("Business meaning:" in note or "Display name:" in note for note in column.notes)
                ),
            }
            for context in contexts
        ],
        "caveats": [caveat for context in contexts for caveat in context.caveats],
        "previous_questions": [question for context in contexts for question in context.previous_questions],
        "conversation": conversation_context or {},
    }


def _findings_relevant_to_question(findings_texts: list[str], question: str) -> list[str]:
    """Filter previous findings to those sharing domain vocabulary with the current question.

    Prevents stale semantic contamination (e.g., entertainment findings leaking into healthcare questions).
    """
    if not findings_texts or not question:
        return findings_texts
    q_tokens = set(question.lower().replace("`", "").split())
    # Generic terms that don't indicate domain relevance
    generic = {
        "the", "a", "an", "is", "are", "was", "were", "of", "in", "to", "for", "by", "with",
        "and", "or", "not", "on", "at", "from", "this", "that", "which", "what", "how",
        "mean", "median", "average", "total", "count", "records", "rows", "across", "among",
        "highest", "lowest", "top", "bottom", "most", "least", "between", "compare",
        "analysis", "dataset", "data", "field", "column", "value", "values", "group",
        "shows", "show", "than", "more", "less", "each", "per", "all", "no", "has", "have",
        "be", "been", "can", "could", "would", "should", "may", "might", "it", "its",
        "overall", "rate", "prevalence", "distribution", "gap", "difference",
    }
    q_domain = q_tokens - generic
    if len(q_domain) < 2:
        return findings_texts
    relevant = []
    for text in findings_texts:
        f_tokens = set(text.lower().replace("`", "").split())
        f_domain = f_tokens - generic
        if q_domain & f_domain:
            relevant.append(text)
    return relevant if relevant else findings_texts[:2]


def _build_conversation_context(
    store: InvestigationStore,
    investigation_id: str,
    active_message_id: str | None = None,
    active_question: str = "",
    limit: int = 12,
) -> dict[str, Any]:
    investigation = store.get_investigation(investigation_id)
    try:
        messages = store.list_investigation_messages(investigation_id, limit=limit)
    except Exception:
        messages = []
    try:
        memory_items = store.list_investigation_memory(investigation_id)
    except Exception:
        memory_items = []
    compact_messages = [
        {
            "message_id": message.message_id,
            "run_id": message.run_id,
            "role": message.role.value,
            "type": message.message_type.value,
            "content": _compact_text(message.content, 900),
            "created_at": message.created_at.isoformat(),
        }
        for message in messages
    ]
    previous_state = investigation.metadata.get("conversation_state") if isinstance(investigation.metadata, dict) else {}
    active_branch_id = str(previous_state.get("active_branch_id") or "") if isinstance(previous_state, dict) else ""
    has_prior_context = bool(active_branch_id or previous_state or list(getattr(investigation, "artifacts", []) or []))
    routing_decision = decide_routing(active_question, has_active_context=has_prior_context)
    context_policy = routing_decision.context_policy
    reset_context = context_policy != ContextPolicy.CONTINUE
    ignore_active_branch = should_ignore_active_branch(routing_decision.question_intent_type)
    if reset_context:
        active_branch_id = ""
    latest_report = investigation.report
    latest_chart_context = {} if reset_context else _latest_chart_context(investigation.artifacts, active_branch_id=active_branch_id)
    memory_payload = [
        {
            "memory_id": item.memory_id,
            "type": item.memory_type.value,
            "status": item.status.value,
            "content": _compact_text(item.content, 420),
            "updated_at": item.updated_at.isoformat(),
        }
        for item in memory_items[:16]
        if item.status.value != "archived"
    ]
    latest_findings = [
        _compact_text(finding.text, 280)
        for finding in investigation.findings[-5:]
        if getattr(finding, "status", None) != "rejected"
    ]
    context_findings = [] if context_policy in {ContextPolicy.RESET_GLOBAL, ContextPolicy.CONVERSATIONAL} else _findings_relevant_to_question(latest_findings, active_question)
    artifact_titles = [
        artifact.title
        for artifact in investigation.artifacts[-8:]
        if getattr(artifact, "visibility", None) != "hidden"
    ]
    thread_state = build_investigation_thread_state(
        {
            "initial_question": investigation.user_question,
            "messages": compact_messages,
            "memory": memory_payload,
            "latest_findings": context_findings,
            "latest_chart_context": latest_chart_context,
            "artifact_titles": artifact_titles,
        }
    )
    resolved_intent = resolve_user_intent(
        active_question,
        has_active_context=bool(latest_chart_context or context_findings or (previous_state if context_policy == ContextPolicy.CONTINUE else {})),
    )
    conversation_state = build_conversation_state(
        previous=_previous_state_for_policy(previous_state, context_policy),
        question=active_question,
        intent=resolved_intent,
        latest_chart_context=latest_chart_context,
        latest_findings=context_findings,
        latest_evidence=[
            getattr(artifact, "title", "")
            for artifact in investigation.artifacts[-8:]
            if context_policy == ContextPolicy.CONTINUE and getattr(artifact, "visibility", None) != "hidden" and getattr(artifact, "title", "")
        ],
        unresolved_questions=[
            item["content"]
            for item in memory_payload
            if item.get("type") == "open_question" and item.get("content")
        ],
    )
    conversation_state_payload = conversation_state.to_payload()
    conversation_state_payload["question_intent_type"] = routing_decision.question_intent_type.value
    conversation_state_payload["branch_action"] = routing_decision.branch_action.value
    conversation_state_payload["context_policy"] = context_policy.value
    _clear_state_for_policy(conversation_state_payload, context_policy)
    if context_policy != ContextPolicy.CONTINUE:
        for stale_key in (
            "active_metric",
            "active_dimension",
            "active_time_axis",
            "active_chart_type",
            "active_analytical_target",
            "active_hypothesis",
        ):
            conversation_state_payload.pop(stale_key, None)
    if isinstance(previous_state, dict) and context_policy == ContextPolicy.CONTINUE:
        for sticky_key in (
            "active_transformation_result",
            "active_adjusted_ranking",
            "active_transformation",
            "active_ranking_scope",
            "active_quality_issue",
            "derived_field",
            "derived_columns",
            "distribution_state",
        ):
            if sticky_key in previous_state and sticky_key not in conversation_state_payload:
                conversation_state_payload[sticky_key] = previous_state.get(sticky_key)
    if isinstance(previous_state, dict) and context_policy not in {ContextPolicy.RESET_GLOBAL, ContextPolicy.CONVERSATIONAL}:
        for schema_key in ("derived_field", "derived_columns"):
            if schema_key in previous_state and schema_key not in conversation_state_payload:
                conversation_state_payload[schema_key] = previous_state.get(schema_key)
    if context_policy == ContextPolicy.CONTINUE and latest_chart_context.get("active_transformation_result"):
        conversation_state_payload["active_transformation_result"] = latest_chart_context.get("active_transformation_result")
        conversation_state_payload["active_transformation"] = latest_chart_context.get("active_transformation") or conversation_state_payload.get("active_transformation", "")
        conversation_state_payload["active_adjusted_ranking"] = latest_chart_context.get("active_adjusted_ranking") or []
    active_message_metadata = _message_metadata_from_list(messages, active_message_id)
    requested_artifact_id = str(active_message_metadata.get("artifact_id") or "") if isinstance(active_message_metadata, dict) else ""
    return {
        "active_message_id": active_message_id,
        "active_message_metadata": active_message_metadata,
        "initial_question": investigation.user_question,
        "resolved_intent": {
            "primary": resolved_intent.primary.value,
            "components": [item.value for item in resolved_intent.components],
            "is_compound": resolved_intent.is_compound,
            "requires_continuity": resolved_intent.requires_continuity,
        },
        "routing_decision": routing_decision.to_payload(),
        "conversation_state": conversation_state_payload,
        "messages": compact_messages,
        "memory": memory_payload,
        "latest_findings": context_findings,
        "latest_report_summary": _compact_text(latest_report.summary or latest_report.answer, 700)
        if latest_report
        else "",
        "latest_chart_context": latest_chart_context,
        "active_analytical_target": conversation_state_payload.get("active_analytical_target") or {},
        "recent_artifacts": _prioritized_artifact_context_payloads(
            investigation.artifacts,
            active_branch_id=active_branch_id,
            requested_artifact_id=requested_artifact_id,
        ),
        "focus": _focus_from_chart_context(latest_chart_context),
        "thread_state": _thread_state_payload(thread_state),
        "artifact_titles": artifact_titles,
    }


def _thread_state_payload(thread_state: Any) -> dict[str, Any]:
    return {
        "active_metric": thread_state.active_metric,
        "active_dimension": thread_state.active_dimension,
        "active_time_axis": thread_state.active_time_axis,
        "active_segments": thread_state.active_segments,
        "active_chart_type": thread_state.active_chart_type,
        "active_business_question": thread_state.active_business_question,
        "active_hypothesis": thread_state.active_hypothesis,
        "recent_findings": thread_state.recent_findings,
        "recent_charts": thread_state.recent_charts,
        "recent_relationships": [
            {
                "relationship_type": item.relationship_type,
                "metric": item.metric,
                "dimension": item.dimension,
                "time_axis": item.time_axis,
                "confidence": item.confidence,
                "rationale": item.rationale,
            }
            for item in thread_state.recent_relationships
        ],
        "unresolved_questions": thread_state.unresolved_questions,
        "suggested_next_questions": thread_state.suggested_next_questions,
    }


RESET_ANALYTICAL_STATE_KEYS = {
    "active_metric",
    "active_dimension",
    "active_time_axis",
    "active_chart_type",
    "active_artifact_id",
    "active_chart",
    "active_distribution_context",
    "active_bin_context",
    "active_topic",
    "active_hypothesis",
    "current_objective",
    "latest_intent",
    "active_branch_id",
    "active_aggregation",
    "active_filters",
    "active_transformation_result",
    "active_adjusted_ranking",
    "active_transformation",
    "active_ranking_scope",
    "active_quality_issue",
    "active_grouped_payload",
    "active_distribution_state",
    "active_artifact_grounding",
    "active_histogram_state",
    "active_comparison_state",
    "stale_semantic_comparison_state",
    "distribution_state",
}


RESET_GLOBAL_EXTRA_STATE_KEYS = {
    "active_dataset_scope",
    "active_dataset_id",
    "active_dataset_ids",
    "active_analytical_target",
    "branch_identity",
    "derived_field",
    "derived_columns",
    "branch_workspace",
}


def _previous_state_for_policy(previous_state: Any, context_policy: ContextPolicy) -> dict[str, Any]:
    previous = dict(previous_state) if isinstance(previous_state, dict) else {}
    if context_policy == ContextPolicy.CONTINUE:
        return previous
    _clear_state_for_policy(previous, context_policy)
    return previous


def _clear_state_for_policy(state: dict[str, Any], context_policy: ContextPolicy) -> None:
    if context_policy == ContextPolicy.CONTINUE:
        return
    keys = set(RESET_ANALYTICAL_STATE_KEYS)
    if context_policy in {ContextPolicy.RESET_GLOBAL, ContextPolicy.CONVERSATIONAL}:
        keys.update(RESET_GLOBAL_EXTRA_STATE_KEYS)
    for key in keys:
        state.pop(key, None)


def _latest_chart_context(artifacts: list[Any], active_branch_id: str = "") -> dict[str, Any]:
    ordered = list(artifacts or [])
    if active_branch_id:
        preferred = [item for item in ordered if _artifact_branch_id(item) == active_branch_id]
        if preferred:
            ordered = preferred
    for artifact in reversed(ordered):
        artifact_type = getattr(getattr(artifact, "artifact_type", None), "value", getattr(artifact, "artifact_type", None))
        if artifact_type != "chart":
            continue
        content = getattr(artifact, "content", None)
        metadata = getattr(artifact, "metadata", None)
        if not isinstance(content, dict):
            content = {}
        if not isinstance(metadata, dict):
            metadata = {}
        nested_metadata = metadata.get("metadata") if isinstance(metadata.get("metadata"), dict) else {}
        chart_type = content.get("chart_type") or metadata.get("chart_type")
        nested_branch_type = metadata.get("branch_type") or nested_metadata.get("branch_type") or ""
        analysis_type = metadata.get("analysis_type") or nested_metadata.get("analysis_type") or ""
        temporal_branch = nested_branch_type in {"trend_analysis", "temporal_decomposition"} or analysis_type in {
            "seasonality",
            "temporal_anomalies",
            "strongest_growth_periods",
            "trend_summary",
        }
        dimension = metadata.get("dimension") or content.get("dimension")
        if chart_type != "line" and not temporal_branch:
            dimension = dimension or content.get("x")
        transformation_types = {"remove_outliers", "median_instead_of_mean", "normalize_by_volume", "exclude_sparse_groups", "stability_check"}
        is_transformation = analysis_type in transformation_types
        return {
            "artifact_id": getattr(artifact, "artifact_id", ""),
            "title": getattr(artifact, "title", ""),
            "dataset_id": metadata.get("dataset_id") or nested_metadata.get("dataset_id") or "",
            "dataset_ids": metadata.get("dataset_ids") or nested_metadata.get("dataset_ids") or [],
            "dataset_scope": metadata.get("dataset_scope") or nested_metadata.get("dataset_scope") or "",
            "chart_type": chart_type,
            "metric": content.get("metric") or metadata.get("metric") or content.get("y"),
            "dimension": dimension,
            "aggregation": metadata.get("aggregation") or _aggregation_from_axis(content.get("y")) or "value",
            "ranking_scope": metadata.get("ranking_scope") or content.get("ranking_scope") or "",
            "row_count": metadata.get("row_count") or content.get("row_count"),
            "filters": content.get("filters") or metadata.get("filters") or [],
            "bins": content.get("bins") or [],
            "displayed_rows": len(content.get("rows") or []) if isinstance(content.get("rows"), list) else None,
            "time_axis": content.get("time_axis") or content.get("timestamp") or metadata.get("timestamp"),
            "branch_type": nested_branch_type,
            "active_transformation": analysis_type if is_transformation else "",
            "active_adjusted_ranking": content.get("rows") if is_transformation else [],
            "active_transformation_result": content.get("transformation_impact") or metadata.get("transformation_impact") or {},
            "base_metric": metadata.get("base_metric") or content.get("base_metric") or metadata.get("metric") or content.get("metric"),
            "base_dimension": metadata.get("base_dimension") or content.get("base_dimension") or dimension,
            "base_aggregation": metadata.get("base_aggregation") or content.get("base_aggregation") or metadata.get("aggregation"),
            "derived_artifact_id": getattr(artifact, "artifact_id", ""),
            "parent_artifact_id": metadata.get("parent_artifact_id") or metadata.get("base_artifact_id") or content.get("parent_artifact_id") or "",
        }
    return {}


def _prioritized_artifact_context_payloads(
    artifacts: list[Any],
    active_branch_id: str = "",
    requested_artifact_id: str = "",
) -> list[dict[str, Any]]:
    visible = list(artifacts or [])[-12:]
    if requested_artifact_id and not any(str(getattr(artifact, "artifact_id", "")) == requested_artifact_id for artifact in visible):
        requested = next((artifact for artifact in artifacts or [] if str(getattr(artifact, "artifact_id", "")) == requested_artifact_id), None)
        if requested is not None:
            visible = [requested, *visible]
    if not active_branch_id:
        return [_artifact_context_payload(artifact) for artifact in visible]
    selected = [artifact for artifact in visible if _artifact_branch_id(artifact) == active_branch_id]
    rest = [artifact for artifact in visible if _artifact_branch_id(artifact) != active_branch_id]
    return [_artifact_context_payload(artifact) for artifact in selected + rest]


def _aggregation_from_axis(axis: Any) -> str:
    value = str(axis or "").strip().lower()
    if value == "total":
        return "sum"
    if value in {"sum", "mean", "median", "count"}:
        return value
    return ""


def _artifact_branch_id(artifact: Any) -> str:
    metadata = getattr(artifact, "metadata", None)
    if not isinstance(metadata, dict):
        return ""
    nested = metadata.get("metadata") if isinstance(metadata.get("metadata"), dict) else {}
    return str(metadata.get("branch_id") or nested.get("branch_id") or "")


def _focus_from_chart_context(chart_context: dict[str, Any]) -> dict[str, Any]:
    if not chart_context:
        return {}
    chart_type = chart_context.get("chart_type")
    if chart_type == "bar":
        analysis_type = "grouped_metric"
    elif chart_type == "line":
        analysis_type = "trend"
    elif chart_type == "histogram":
        analysis_type = "distribution"
    elif chart_type == "scatter":
        analysis_type = "correlation"
    else:
        analysis_type = chart_type
    return {
        "active_metric": chart_context.get("metric"),
        "active_dimension": "" if chart_type == "line" else chart_context.get("dimension"),
        "active_chart": chart_context.get("title"),
        "active_analysis_type": analysis_type,
        "active_branch_type": chart_context.get("branch_type") or ("trend_analysis" if chart_type == "line" else "grouped_comparison" if chart_type == "bar" else analysis_type),
        "active_time_axis": chart_context.get("time_axis") if chart_type == "line" else None,
    }


def _artifact_context_payload(artifact: Any) -> dict[str, Any]:
    artifact_type = getattr(getattr(artifact, "artifact_type", None), "value", getattr(artifact, "artifact_type", None))
    return {
        "artifact_id": getattr(artifact, "artifact_id", ""),
        "artifact_type": artifact_type,
        "title": getattr(artifact, "title", ""),
        "content": getattr(artifact, "content", None),
        "metadata": getattr(artifact, "metadata", {}) if isinstance(getattr(artifact, "metadata", {}), dict) else {},
        "created_at": getattr(getattr(artifact, "created_at", None), "isoformat", lambda: "")(),
    }


def _message_metadata_from_list(messages: list[InvestigationMessage], active_message_id: str | None) -> dict[str, Any]:
    if not active_message_id:
        return {}
    for message in messages:
        if getattr(message, "message_id", "") == active_message_id:
            metadata = getattr(message, "metadata", {})
            return metadata if isinstance(metadata, dict) else {}
    return {}


def _message_selected_dataset_ids(message: InvestigationMessage | None) -> list[str]:
    metadata = getattr(message, "metadata", {}) if message is not None else {}
    if not isinstance(metadata, dict):
        return []
    values = metadata.get("dataset_ids") or metadata.get("data_source_ids") or []
    if isinstance(values, str):
        values = [values]
    selected = [str(item) for item in values if str(item).strip()] if isinstance(values, list) else []
    single = str(metadata.get("dataset_id") or metadata.get("data_source_id") or "").strip()
    if single:
        selected.insert(0, single)
    return _unique_ids(selected)


def _activate_branch_from_message_metadata(store: InvestigationStore, investigation_id: str, message: InvestigationMessage | None) -> None:
    metadata = getattr(message, "metadata", {}) if message is not None else {}
    if not isinstance(metadata, dict):
        return
    branch_id = str(metadata.get("branch_id") or "")
    if not branch_id:
        return
    try:
        investigation = store.get_investigation(investigation_id)
        updated = activate_branch(investigation, branch_id)
        updater = getattr(store, "update_investigation_metadata", None)
        if callable(updater):
            updater(investigation_id, updated)
    except Exception:
        return


def _summarize_run_result(investigation: Any) -> str:
    if getattr(investigation, "report", None):
        report = investigation.report
        text = report.summary or report.answer or report.content
        if text:
            return _open_ended_summary(str(text), investigation)
    findings = [finding.text for finding in getattr(investigation, "findings", [])[-3:]]
    if findings:
        return "Analytical takeaway: " + " ".join(_compact_text(item, 240) for item in findings)
    artifacts_count = len(getattr(investigation, "artifacts", []) or [])
    if artifacts_count:
        chart_titles = [
            getattr(artifact, "title", "")
            for artifact in getattr(investigation, "artifacts", [])[-5:]
            if getattr(getattr(artifact, "artifact_type", None), "value", getattr(artifact, "artifact_type", None)) == "chart"
            and getattr(artifact, "title", "")
        ]
        if chart_titles:
            return f"Chart evidence is available: {chart_titles[-1]}."
    # ── Terminal decision check: extract from latest run output ──
    # If the latest run has a terminal decision (e.g. semantic_incompatibility),
    # return that message instead of generic "no evidence" fallback.
    _terminal = _extract_terminal_decision_from_runs(investigation)
    if _terminal:
        return _terminal
    return "I do not have a fresh row-level result for this question yet. Anchor the next check to a metric, segment, trend, anomaly, chart, or evidence gap so the answer can stay tied to the investigation."


def _extract_terminal_decision_from_runs(investigation: Any) -> str | None:
    """Check latest runs for terminal decisions like semantic incompatibility.

    When a compatibility check blocks execution, the run output contains
    the refusal message but no report/findings/artifacts are stored.
    This function extracts the terminal message directly from the run output.
    """
    runs = getattr(investigation, "runs", None) or []
    if not runs:
        return None
    # Check the latest run(s) for terminal decisions
    for run in reversed(runs[-3:]):
        output = getattr(run, "output", None)
        if not isinstance(output, dict):
            continue
        trace = output.get("trace_metadata") or {}
        if not isinstance(trace, dict):
            continue
        analysis_type = str(trace.get("analysis_type") or "")
        # Terminal decision types that should never be replaced with "no evidence"
        if analysis_type in (
            "semantic_incompatibility",
            "hard_stop",
            "hard_stop_missing_fields",
            "insufficient_dataset_scope",
        ):
            summary = str(output.get("summary") or output.get("final_answer") or "").strip()
            if summary:
                return summary
    return None


def _is_execution_context_unavailable_summary(summary: str, investigation: Any | None = None) -> bool:
    normalized = " ".join(str(summary or "").casefold().split())
    if "executable dataset context is unavailable" in normalized or "raw rows are not attached" in normalized:
        return True
    report = getattr(investigation, "report", None) if investigation is not None else None
    metadata = getattr(report, "metadata", {}) if report is not None else {}
    trace = metadata.get("trace_metadata") if isinstance(metadata, dict) and isinstance(metadata.get("trace_metadata"), dict) else {}
    return bool(trace.get("execution_context_unavailable"))


def _looks_like_state_clarification(text: str) -> bool:
    normalized = " ".join(str(text or "").lower().split())
    return (
        "ты имеешь в виду продолжить текущую ветку" in normalized
        or "или начать новую проверку" in normalized
        or "do you want to continue the current" in normalized
        or "or start a new check" in normalized
    )


def _output_summary(output: Any) -> str:
    if not isinstance(output, dict):
        return ""
    for key in ("final_answer", "summary", "result_preview"):
        value = str(output.get(key) or "").strip()
        if value:
            return value
    report = output.get("structured_report")
    if isinstance(report, dict):
        return str(report.get("summary") or "").strip()
    return ""


def _open_ended_summary(text: str, investigation: Any | None = None) -> str:
    normalized = " ".join(str(text or "").lower().split())
    terminal_phrases = (
        "done",
        "analysis completed",
        "i finished the analysis",
        "the analysis is complete",
        "main takeaway is available above",
        "available above",
        "i found new analytical material",
        "you can keep exploring",
        "the investigation has been updated",
        "continue with another follow-up",
        "supporting findings and visuals nearby",
    )
    if normalized in terminal_phrases or any(phrase in normalized for phrase in terminal_phrases[1:]):
        return _analytical_summary_from_state(investigation)
    return text


def _analytical_summary_from_state(investigation: Any | None) -> str:
    findings = list(getattr(investigation, "findings", []) or []) if investigation is not None else []
    if findings:
        latest = findings[-1]
        text = getattr(latest, "text", "") or getattr(latest, "title", "")
        metadata = getattr(latest, "metadata", {}) or {}
        conclusion = metadata.get("conclusion") if isinstance(metadata, dict) else ""
        limitation = metadata.get("limitation") if isinstance(metadata, dict) else ""
        validation = metadata.get("recommended_validation") if isinstance(metadata, dict) else ""
        parts = [str(conclusion or text).strip()]
        if limitation:
            parts.append(f"Limitation: {limitation}")
        if validation:
            parts.append(f"Validation: {validation}")
        return " ".join(part for part in parts if part)
    artifacts = list(getattr(investigation, "artifacts", []) or []) if investigation is not None else []
    for artifact in reversed(artifacts):
        artifact_type = getattr(getattr(artifact, "artifact_type", None), "value", getattr(artifact, "artifact_type", None))
        if artifact_type == "chart":
            title = getattr(artifact, "title", "") or "the latest chart"
            return f"{title} is available as chart evidence. Use it to compare the strongest groups, check whether the pattern is stable, and decide what needs validation next."
    # Check runs for terminal decisions before falling back to generic message
    _terminal = _extract_terminal_decision_from_runs(investigation)
    if _terminal:
        return _terminal
    return "I do not have a fresh row-level result for this question yet. The next useful check should stay tied to the current metric, segment, chart, anomaly, or evidence gap."


def is_semantically_redundant_response(candidate: str, previous: str | list[str], state: dict[str, Any] | None = None) -> bool:
    """Detect repeated assistant prose without requiring exact string equality."""

    previous_messages = previous if isinstance(previous, list) else [previous]
    return conversation_redundant_response(candidate, previous_messages, state)


def _is_chart_request(question: str) -> bool:
    text = " ".join(str(question or "").lower().replace("_", " ").split())
    return any(
        marker in text
        for marker in (
            "chart",
            "graph",
            "plot",
            "visualize",
            "line chart",
            "график",
            "построй график",
            "нарисуй",
            "визуализируй",
            "диаграмма",
            "линия",
            "тренд",
        )
    )


def _semantic_signature(value: str) -> str:
    text = " ".join(str(value or "").lower().split())
    cleaned = []
    for char in text:
        if char.isalnum() or char.isspace() or char == "_":
            cleaned.append(char)
        else:
            cleaned.append(" ")
    tokens = " ".join("".join(cleaned).split())
    return tokens


def _non_redundant_follow_up_response(question: str, investigation: Any | None, repeated_summary: str, *, df: Any = None) -> str:
    if execution_required_for_question(question, df=df):
        return repeated_summary
    routing_decision = decide_routing(question, has_active_context=True)
    if should_ignore_active_branch(routing_decision.question_intent_type):
        return repeated_summary
    chart_context = _latest_chart_context(list(getattr(investigation, "artifacts", []) or [])) if investigation is not None else {}
    metric = str(chart_context.get("metric") or "").strip()
    dimension = str(chart_context.get("dimension") or "").strip()
    state = {}
    if investigation is not None and isinstance(getattr(investigation, "metadata", None), dict):
        state = investigation.metadata.get("conversation_state") or {}
    transformed_answer = _transformed_ranking_follow_up_response(question, state)
    if transformed_answer:
        return transformed_answer
    duplicate_repeat = _duplicate_repeat_answer(question, df)
    if duplicate_repeat:
        return duplicate_repeat
    if is_evidence_followup(question) and isinstance(state, dict) and state.get("active_analytical_target"):
        resolution = resolve_evidence_subject(question, state.get("active_analytical_target"), branch_state=state)
        if resolution.target:
            rows = _active_target_rows_for_investigation(resolution.target.to_payload(), investigation)
            return evidence_response_text(question, resolution.target, artifact_rows=rows)
    deterministic = _deterministic_followup_output(question, df, state, investigation)
    if deterministic:
        return deterministic
    findings = list(getattr(investigation, "findings", []) or []) if investigation is not None else []
    latest_finding = ""
    if findings:
        finding = findings[-1]
        metadata = getattr(finding, "metadata", {}) or {}
        latest_finding = str(metadata.get("conclusion") or getattr(finding, "text", "") or getattr(finding, "title", "")).strip()
    normalized_question = str(question or "").lower()
    if "evidence" in normalized_question or "доказ" in normalized_question or "подтверж" in normalized_question:
        target = latest_finding or (f"`{metric}` by `{dimension}`" if metric and dimension else "the latest conclusion")
        return (
            f"No concrete supporting table or chart is available yet for {target}. "
            "A useful evidence answer needs a computed rank shift, group comparison, quality check, or hypothesis result tied to the same metric."
        )
    if metric and dimension:
        ranked_answer = _ranked_group_follow_up_response(question, df, metric, dimension, chart_context)
        if ranked_answer:
            return ranked_answer
        return (
            f"The current evidence centers on `{metric}` by `{dimension}`. "
            "A useful next check is to validate the leading groups with record volume, outliers, and time stability."
        )
    if latest_finding:
        return (
            f"The strongest stored finding still points to this issue: {latest_finding}. "
            "Next, run a direct comparison, anomaly check, transformation, or validation tied to the same metric."
        )
    return (
        "This follow-up needs a more specific analytical anchor before I can add a new conclusion. "
        "Name the metric, group, chart, quality issue, or hypothesis you want to validate next."
    )


def _deterministic_followup_output(question: str, df: Any, state: dict[str, Any], investigation: Any | None) -> str:
    if df is None:
        return ""
    findings = []
    if investigation is not None:
        for finding in list(getattr(investigation, "findings", []) or [])[-5:]:
            metadata = getattr(finding, "metadata", {}) or {}
            text = str(metadata.get("conclusion") or getattr(finding, "text", "") or getattr(finding, "title", "")).strip()
            if text:
                findings.append(text)
    data_context = {
        "conversation_context": {
            "conversation_state": state if isinstance(state, dict) else {},
            "latest_findings": findings,
        }
    }
    output = deterministic_investigation_fallback(question, df, data_context=data_context)
    if isinstance(output, dict):
        summary = str(output.get("summary") or "").strip()
        if summary:
            return summary
    return ""


def _duplicate_repeat_answer(question: str, df: Any) -> str:
    normalized = " ".join(str(question or "").casefold().split())
    if not any(marker in normalized for marker in ("duplicate", "duplicates", "duplicated", "дублик", "повтор")):
        return ""
    if df is None or not hasattr(df, "duplicated") or not hasattr(df, "columns"):
        return ""
    try:
        duplicate_rows = int(df.duplicated(keep=False).sum())
        duplicate_patterns = int(df.duplicated(keep="first").sum())
        order_col = next((str(col) for col in df.columns if "order" in str(col).casefold() and "id" in str(col).casefold()), "")
        order_note = ""
        if order_col and order_col in df.columns:
            repeated_ids = int(df[order_col].duplicated(keep=False).sum())
            distinct_repeated = int(df.loc[df[order_col].duplicated(keep=False), order_col].nunique())
            order_note = (
                f" `{order_col}` has {distinct_repeated:,} repeated ID values across {repeated_ids:,} rows, "
                "which may be normal line items rather than duplicate orders."
            )
        return (
            f"The duplicate picture is unchanged: exact duplicate rows affect {duplicate_rows:,} rows "
            f"({duplicate_patterns:,} removable repeated row patterns).{order_note} "
            "Next, check whether those duplicates concentrate in the active grouping or another relevant schema dimension."
        )
    except Exception:
        return ""


def _transformed_ranking_follow_up_response(question: str, state: dict[str, Any]) -> str | None:
    if not isinstance(state, dict):
        return None
    transformed = transformation_state_from_payload(state.get("active_transformation_result"))
    if not transformed or not transformed.adjusted_ranking:
        transformed = transformation_state_from_payload(
            {
                "metric": state.get("active_metric") or "",
                "dimension": state.get("active_dimension") or "",
                "transformation_type": state.get("active_transformation") or "",
                "adjusted_ranking": state.get("active_adjusted_ranking") or [],
                "ranking_scope": state.get("active_ranking_scope") or "",
            }
        )
    if not transformed or not transformed.adjusted_ranking:
        return None
    text = " ".join(str(question or "").lower().split())
    if not any(marker in text for marker in ("strongest", "leader", "leaders", "top", "best", "remain", "changed", "самые", "сильн", "лидер", "остаются", "остались", "измен")):
        return None
    metric = transformed.metric or str(state.get("active_metric") or "metric")
    dimension = transformed.dimension or str(state.get("active_dimension") or "group")
    if any(marker in text for marker in ("what changed", "changed", "after filtering", "became unreliable", "измен")) and transformed.comparison_rows:
        execution = transformation_from_payload(transformed.to_payload())
        if execution:
            return synthesize_transformation_change(execution)
    rows = transformed.adjusted_ranking[:5]
    value_key = _first_existing_key_dict(rows[0], ("adjusted_mean", "mean", "median", "total")) if rows else ""
    count_key = _first_existing_key_dict(rows[0], ("count", "adjusted_count", "record_count")) if rows else ""
    leaders = []
    for row in rows:
        value = row.get(dimension)
        score = row.get(value_key) if value_key else None
        count = row.get(count_key) if count_key else None
        bit = f"`{value}`"
        if score is not None:
            bit += f" ({value_key or 'value'} {float(score):.2f}"
            if count is not None:
                bit += f", n={int(float(count))}"
            bit += ")"
        leaders.append(bit)
    return (
        f"After filtering, the adjusted ranking for `{metric}` by `{dimension}` has these strongest groups: {', '.join(leaders)}. "
        "These are adjusted leaders, and the tiny-n groups still need a median plus minimum-sample check before they are treated as robust."
    )


def _first_existing_key_dict(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in row:
            return key
    return ""


def _duplicate_follow_up_deepening(question: str, investigation: Any | None, *, df: Any = None) -> str | None:
    chart_context = _latest_chart_context(list(getattr(investigation, "artifacts", []) or [])) if investigation is not None else {}
    metric = str(chart_context.get("metric") or "").strip()
    dimension = str(chart_context.get("dimension") or "").strip()
    state = {}
    if investigation is not None and isinstance(getattr(investigation, "metadata", None), dict):
        state = investigation.metadata.get("conversation_state") or {}
    transformed_answer = _transformed_ranking_follow_up_response(question, state)
    if transformed_answer:
        return transformed_answer
    if not metric or not dimension or df is None or not hasattr(df, "columns") or metric not in df.columns or dimension not in df.columns:
        return None
    ranked = _ranked_group_follow_up_response(question, df, metric, dimension, chart_context)
    if not ranked:
        return None
    try:
        grouped = (
            df.groupby(dimension, dropna=False)[metric]
            .agg(mean="mean", median="median", count="count", min="min", max="max")
            .reset_index()
            .sort_values(["mean", "count"], ascending=[False, False])
        )
    except Exception:
        return None
    if grouped.empty:
        return None
    top = grouped.head(5)
    sparse = top[top["count"] <= 2]
    stable = top[top["count"] > 2]
    language = ResponseLanguagePolicy.from_message(question, protected_terms=[metric, dimension])
    leaders = ", ".join(f"`{row[dimension]}`" for _, row in top.head(3).iterrows())
    stable_leaders = ", ".join(f"`{row[dimension]}`" for _, row in stable.head(3).iterrows())
    sparse_leaders = ", ".join(f"`{row[dimension]}`" for _, row in sparse.head(3).iterrows())
    if language.is_russian:
        reliability = (
            f"Более надежная часть ответа: {stable_leaders}." if stable_leaders
            else "Надежных лидеров с нормальным числом строк в верхушке почти нет."
        )
        fragility = f"Хрупкая часть: {sparse_leaders} выглядят сильными, но выборка маленькая." if sparse_leaders else ""
        return (
            f"Коротко: рейтинг тот же, лидируют {leaders}. "
            f"Но главный вывод не в повторе списка, а в надежности: {reliability} {fragility} "
            "Следующая проверка по этой же ветке — убрать extreme orders или перейти на median, чтобы понять, кто остается сильным без влияния отдельных крупных записей."
        )
    reliability = (
        f"The better-supported leaders are {stable_leaders}." if stable_leaders
        else "The top leaders have very thin support."
    )
    fragility = f"The fragile part is {sparse_leaders}: they rank high but have tiny samples." if sparse_leaders else ""
    return (
        f"Same ranking: {leaders} lead. "
        f"The useful extra point is reliability: {reliability} {fragility} "
        "The next check in this same thread is to remove extreme records or use median to see which leaders survive."
    )


def _active_target_rows_for_investigation(active_target: dict[str, Any], investigation: Any | None) -> list[dict[str, Any]]:
    artifacts = list(getattr(investigation, "artifacts", []) or []) if investigation is not None else []
    target_metric = str(active_target.get("metric") or "")
    target_dimension = str(active_target.get("dimension") or "")
    target_mechanism = str(active_target.get("active_mechanism") or "")
    for artifact in reversed(artifacts):
        content = getattr(artifact, "content", None)
        metadata = getattr(artifact, "metadata", None)
        if isinstance(content, dict):
            content_dict = content
            rows = content.get("rows") if isinstance(content.get("rows"), list) else []
        else:
            content_dict = {}
            rows = content if isinstance(content, list) else []
        metadata = metadata if isinstance(metadata, dict) else {}
        nested_metadata = metadata.get("metadata") if isinstance(metadata.get("metadata"), dict) else {}
        impact = content_dict.get("transformation_impact") if isinstance(content_dict.get("transformation_impact"), dict) else metadata.get("transformation_impact")
        if isinstance(impact, dict):
            metric = str(impact.get("metric") or metadata.get("metric") or nested_metadata.get("metric") or content_dict.get("metric") or "")
            dimension = str(impact.get("dimension") or metadata.get("dimension") or nested_metadata.get("dimension") or content_dict.get("dimension") or content_dict.get("x") or "")
            if (not target_metric or not metric or metric == target_metric) and (not target_dimension or not dimension or dimension == target_dimension):
                comparison_rows = impact.get("comparison_rows")
                if isinstance(comparison_rows, list) and comparison_rows:
                    return [row for row in comparison_rows if isinstance(row, dict)]
        if not rows or not all(isinstance(row, dict) for row in rows):
            continue
        metric = str(metadata.get("metric") or nested_metadata.get("metric") or content_dict.get("metric") or "")
        dimension = str(metadata.get("dimension") or nested_metadata.get("dimension") or content_dict.get("dimension") or content_dict.get("x") or "")
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
        return rows
    return []


def _ranked_group_follow_up_response(
    question: str,
    df: Any,
    metric: str,
    dimension: str,
    chart_context: dict[str, Any],
) -> str | None:
    if df is None or not hasattr(df, "columns") or metric not in df.columns or dimension not in df.columns:
        return None
    normalized_question = " ".join(str(question or "").lower().split())
    ranking_markers = (
        "strongest",
        "highest",
        "leaders",
        "top",
        "best",
        "самые сильные",
        "сильн",
        "лидер",
        "лучшие",
        "топ",
        "выше",
    )
    if not any(marker in normalized_question for marker in ranking_markers):
        return None
    try:
        grouped = (
            df.groupby(dimension, dropna=False)[metric]
            .agg(mean="mean", median="median", total="sum", count="count", min="min", max="max")
            .reset_index()
            .sort_values(["mean", "count"], ascending=[False, False])
        )
    except Exception:
        return None
    if grouped.empty:
        return None
    top = grouped.head(5)
    leader = top.iloc[0]
    sparse = top[top["count"] <= 2]
    stable = top[top["count"] > 2]
    widest = grouped.assign(spread=grouped["max"] - grouped["min"]).sort_values("spread", ascending=False).iloc[0]
    displayed_rows = chart_context.get("displayed_rows")
    scope = str(chart_context.get("ranking_scope") or (
        f"Top {displayed_rows} displayed groups by average `{metric}`" if displayed_rows else f"all `{dimension}` groups by average `{metric}`"
    )).rstrip(".")
    language = ResponseLanguagePolicy.from_message(question, protected_terms=[metric, dimension])
    if language.is_russian:
        top_bits = [
            f"`{row[dimension]}`: avg {float(row['mean']):.2f}, median {float(row['median']):.2f}, {_ru_rows_short(int(row['count']))}"
            for _, row in top.iterrows()
        ]
        leader_count = int(leader["count"])
        caveat = (
            f"Но лидер `{leader[dimension]}` держится на {_ru_rows(leader_count)}, поэтому силу надо читать вместе с надежностью выборки. "
            if leader_count <= 2
            else f"Лидер `{leader[dimension]}` выглядит устойчивее, потому что у него {_ru_rows(leader_count)}. "
        )
        sparse_text = ""
        if not sparse.empty:
            sparse_text = "Верхушка частично хрупкая: " + ", ".join(f"`{row[dimension]}` ({int(row['count'])})" for _, row in sparse.head(3).iterrows()) + " имеют мало строк. "
        stable_text = ""
        if not stable.empty:
            stable_text = "Более надежные сильные группы среди top выглядят так: " + ", ".join(f"`{row[dimension]}`" for _, row in stable.head(3).iterrows()) + ". "
        return (
            f"Самые сильные `{dimension}` по среднему `{metric}`: {', '.join(top_bits)}. "
            f"График сейчас показывает {scope}. "
            f"{caveat}{sparse_text}{stable_text}"
            f"Самый широкий внутренний разброс у `{widest[dimension]}`: {float(widest['min']):.2f}–{float(widest['max']):.2f}, "
            "поэтому часть разницы может идти не от самого города, а от отдельных крупных заказов или смешения подгрупп."
        )
    top_bits = [
        f"`{row[dimension]}`: avg {float(row['mean']):.2f}, median {float(row['median']):.2f}, {int(row['count'])} {_en_rows(int(row['count']))}"
        for _, row in top.iterrows()
    ]
    caveat = (
        f"The leader `{leader[dimension]}` is based on only {int(leader['count'])} rows, so its strength is not very stable yet. "
        if int(leader["count"]) <= 2
        else f"The leader `{leader[dimension]}` is more stable because it has {int(leader['count'])} rows. "
    )
    sparse_text = ""
    if not sparse.empty:
        sparse_text = "The top is partly fragile: " + ", ".join(f"`{row[dimension]}` ({int(row['count'])})" for _, row in sparse.head(3).iterrows()) + " have very little support. "
    stable_text = ""
    if not stable.empty:
        stable_text = "The stronger supported leaders in the top set are " + ", ".join(f"`{row[dimension]}`" for _, row in stable.head(3).iterrows()) + ". "
    return (
        f"Same ranking detail with reliability context: the strongest `{dimension}` groups by average `{metric}` are {', '.join(top_bits)}. "
        f"The current chart shows {scope}. "
        f"{caveat}{sparse_text}{stable_text}"
        f"The widest within-group spread is `{widest[dimension]}` ({float(widest['min']):.2f} to {float(widest['max']):.2f}), "
        "so part of the apparent gap may come from individual large records or subgroup mix."
    )


def _ru_rows(count: int) -> str:
    if count % 10 == 1 and count % 100 != 11:
        suffix = "строке"
    elif count % 10 in {2, 3, 4} and count % 100 not in {12, 13, 14}:
        suffix = "строках"
    else:
        suffix = "строках"
    return f"{count} {suffix}"


def _ru_rows_short(count: int) -> str:
    if count % 10 == 1 and count % 100 != 11:
        suffix = "строка"
    elif count % 10 in {2, 3, 4} and count % 100 not in {12, 13, 14}:
        suffix = "строки"
    else:
        suffix = "строк"
    return f"{count} {suffix}"


def _en_rows(count: int) -> str:
    return "row" if count == 1 else "rows"


def _is_self_sufficient_dataset_scan(question: str, summary: str) -> bool:
    normalized_question = str(question or "").lower()
    normalized_summary = str(summary or "").lower()
    asks_dataset_scan = any(
        marker in normalized_question
        for marker in ("что ты можешь сказать", "dependencies", "dependency", "relationships", "завис", "связ")
    )
    concrete_markers = ("`sales`", "highest", "lowest", "median", "mean", "max", "correlation", "varies", "строк", "колон", "диапазон")
    generic_markers = ("active conclusion", "to move beyond", "useful next step would be", "key findings should")
    return asks_dataset_scan and any(marker in normalized_summary for marker in concrete_markers) and not any(
        marker in normalized_summary for marker in generic_markers
    )


def _persist_conversation_state(
    store: InvestigationStore,
    investigation_id: str,
    *,
    question: str,
    run_id: str,
) -> None:
    try:
        investigation = store.get_investigation(investigation_id)
        latest_chart = _latest_chart_context(investigation.artifacts)
        latest_findings = [
            _compact_text(finding.text, 320)
            for finding in investigation.findings[-8:]
            if getattr(finding, "status", None) != "rejected"
        ]
        latest_evidence = [
            getattr(artifact, "title", "")
            for artifact in investigation.artifacts[-8:]
            if getattr(artifact, "visibility", None) != "hidden" and getattr(artifact, "title", "")
        ]
        previous = investigation.metadata.get("conversation_state") if isinstance(investigation.metadata, dict) else {}
        intent = resolve_user_intent(question, has_active_context=bool(latest_chart or latest_findings or previous))
        routing_decision = decide_routing(question, has_active_context=bool(previous))
        context_policy = routing_decision.context_policy
        reset_context = context_policy != ContextPolicy.CONTINUE
        ignore_active_branch = should_ignore_active_branch(routing_decision.question_intent_type)
        if reset_context:
            latest_chart = {}
            latest_findings = []
            latest_evidence = []
        else:
            latest_findings = _findings_relevant_to_question(latest_findings, question)
        state = build_conversation_state(
            previous=_previous_state_for_policy(previous, context_policy),
            question=question,
            intent=intent,
            latest_chart_context=latest_chart,
            latest_findings=latest_findings,
            latest_evidence=latest_evidence,
        )
        metadata = dict(investigation.metadata or {})
        latest_finding_obj = next(
            (finding for finding in reversed(investigation.findings or []) if getattr(finding, "status", None) != "rejected"),
            None,
        )
        if context_policy in {ContextPolicy.RESET_GLOBAL, ContextPolicy.CONVERSATIONAL}:
            latest_finding_obj = None
        report_summary = ""
        trace_metadata: dict[str, Any] = {}
        if investigation.report:
            report_summary = _compact_text(investigation.report.summary or investigation.report.answer or "", 700)
            report_meta = getattr(investigation.report, "metadata", {}) or {}
            trace_metadata = report_meta.get("trace_metadata") if isinstance(report_meta.get("trace_metadata"), dict) else {}
        if trace_metadata.get("execution_context_unavailable"):
            metadata = dict(investigation.metadata or {})
            previous_state = previous if isinstance(previous, dict) else {}
            failed_state = dict(previous_state)
            failed_state.update(
                {
                    "last_error_type": "execution_context_unavailable",
                    "last_failed_plan": trace_metadata.get("last_failed_plan") or trace_metadata.get("query_plan") or {},
                    "last_failed_dataset_id": trace_metadata.get("data_source_id") or trace_metadata.get("dataset_id") or "",
                    "last_failed_question": question,
                    "last_run_id": run_id,
                }
            )
            metadata["conversation_state"] = failed_state
            metadata["last_error_type"] = "execution_context_unavailable"
            metadata["last_failed_plan"] = failed_state["last_failed_plan"]
            metadata["last_failed_dataset_id"] = failed_state["last_failed_dataset_id"]
            updater = getattr(store, "update_investigation_metadata", None)
            if callable(updater):
                updater(investigation_id, metadata)
            else:
                investigation.metadata = metadata
            return
        if not trace_metadata and latest_finding_obj is not None:
            finding_meta = getattr(latest_finding_obj, "metadata", {}) or {}
            trace_metadata = {
                key: finding_meta.get(key)
                for key in (
                    "analysis_type",
                    "active_branch_type",
                    "metric",
                    "dimension",
                    "time_axis",
                    "timestamp",
                    "matched_category_value",
                    "hypothesis_mechanism",
                    "active_quality_issue",
                )
                if finding_meta.get(key) not in (None, "")
            }
        if str(trace_metadata.get("analysis_type") or "") == "clarification_needed" and isinstance(previous, dict) and isinstance(previous.get("active_analytical_target"), dict):
            active_target = dict(previous.get("active_analytical_target") or {})
        else:
            active_target = build_active_target(
                question=question,
                state=state.to_payload(),
                trace_metadata=trace_metadata,
                finding=latest_finding_obj,
                chart=latest_chart,
                report_summary=report_summary,
                run_id=run_id,
            ).to_payload()
        branch_identity = BranchIdentity(
            metric=str(active_target.get("metric") or ""),
            dimension=str(active_target.get("dimension") or ""),
            branch_type=str(active_target.get("branch_type") or ""),
            active_entities=[str(item) for item in active_target.get("active_entities", []) if str(item).strip()],
            active_hypothesis=str(active_target.get("hypothesis") or ""),
            active_transformation=str(active_target.get("active_transformation") or ""),
        ).to_payload()
        metadata["conversation_state"] = {
            **state.to_payload(),
            "last_run_id": run_id,
            "active_analytical_target": active_target,
            "branch_identity": branch_identity,
            "question_intent_type": routing_decision.question_intent_type.value,
            "branch_action": routing_decision.branch_action.value,
            "context_policy": context_policy.value,
        }
        _clear_state_for_policy(metadata["conversation_state"], context_policy)
        if latest_chart:
            metadata["conversation_state"]["active_artifact_id"] = latest_chart.get("artifact_id") or ""
            metadata["conversation_state"]["active_dataset_id"] = latest_chart.get("dataset_id") or metadata["conversation_state"].get("active_dataset_id")
            metadata["conversation_state"]["active_dataset_ids"] = latest_chart.get("dataset_ids") or metadata["conversation_state"].get("active_dataset_ids") or []
            metadata["conversation_state"]["active_dataset_scope"] = latest_chart.get("dataset_scope") or metadata["conversation_state"].get("active_dataset_scope")
            metadata["conversation_state"]["active_metric"] = latest_chart.get("metric") or metadata["conversation_state"].get("active_metric")
            metadata["conversation_state"]["active_dimension"] = latest_chart.get("dimension") or metadata["conversation_state"].get("active_dimension")
            metadata["conversation_state"]["active_aggregation"] = latest_chart.get("aggregation") or metadata["conversation_state"].get("active_aggregation")
            metadata["conversation_state"]["active_filters"] = latest_chart.get("filters") or []
            metadata["conversation_state"]["active_chart_type"] = latest_chart.get("chart_type") or metadata["conversation_state"].get("active_chart_type")
            if latest_chart.get("chart_type") == "histogram":
                metadata["conversation_state"]["active_distribution_context"] = {
                    "artifact_id": latest_chart.get("artifact_id") or "",
                    "metric": latest_chart.get("metric") or "",
                    "dimension": latest_chart.get("dimension") or "",
                    "filters": latest_chart.get("filters") or [],
                    "chart_type": latest_chart.get("chart_type") or "",
                }
        if isinstance(previous, dict) and previous.get("active_branch_id") and context_policy == ContextPolicy.CONTINUE:
            metadata["conversation_state"]["active_branch_id"] = previous.get("active_branch_id")
            metadata["conversation_state"]["branch_workspace"] = previous.get("branch_workspace") or metadata.get("branch_workspace")
        if isinstance(previous, dict) and context_policy == ContextPolicy.CONTINUE:
            for sticky_key in (
                "active_transformation_result",
                "active_adjusted_ranking",
                "active_transformation",
                "active_ranking_scope",
                "active_quality_issue",
            ):
                if sticky_key in previous and sticky_key not in metadata["conversation_state"]:
                    metadata["conversation_state"][sticky_key] = previous.get(sticky_key)
        metadata["active_analytical_target"] = active_target
        metadata["branch_identity"] = branch_identity
        if trace_metadata.get("active_quality_issue"):
            metadata["conversation_state"]["active_quality_issue"] = trace_metadata.get("active_quality_issue")
        if isinstance(trace_metadata.get("derived_field"), dict):
            metadata["conversation_state"]["derived_field"] = trace_metadata.get("derived_field")
        if context_policy == ContextPolicy.CONTINUE and isinstance(previous, dict) and isinstance(previous.get("derived_field"), dict) and "derived_field" not in metadata["conversation_state"]:
            metadata["conversation_state"]["derived_field"] = previous.get("derived_field")
        if isinstance(trace_metadata.get("derived_columns"), list):
            metadata["conversation_state"]["derived_columns"] = trace_metadata.get("derived_columns")
        if context_policy == ContextPolicy.CONTINUE and isinstance(previous, dict) and isinstance(previous.get("derived_columns"), list) and "derived_columns" not in metadata["conversation_state"]:
            metadata["conversation_state"]["derived_columns"] = previous.get("derived_columns")
        if context_policy not in {ContextPolicy.RESET_GLOBAL, ContextPolicy.CONVERSATIONAL} and isinstance(previous, dict):
            if isinstance(previous.get("derived_field"), dict) and "derived_field" not in metadata["conversation_state"]:
                metadata["conversation_state"]["derived_field"] = previous.get("derived_field")
            if isinstance(previous.get("derived_columns"), list) and "derived_columns" not in metadata["conversation_state"]:
                metadata["conversation_state"]["derived_columns"] = previous.get("derived_columns")
        if isinstance(trace_metadata.get("distribution_state"), dict):
            metadata["conversation_state"]["distribution_state"] = trace_metadata.get("distribution_state")
        if context_policy == ContextPolicy.CONTINUE and isinstance(previous, dict) and isinstance(previous.get("distribution_state"), dict) and "distribution_state" not in metadata["conversation_state"]:
            metadata["conversation_state"]["distribution_state"] = previous.get("distribution_state")
        plan_payload = trace_metadata.get("query_plan") if isinstance(trace_metadata.get("query_plan"), dict) else {}
        if plan_payload:
            plan_payload = dict(plan_payload)
            dataset_resolution = trace_metadata.get("dataset_resolution") if isinstance(trace_metadata.get("dataset_resolution"), dict) else {}
            dataset_ids = dataset_resolution.get("selected_dataset_ids") if isinstance(dataset_resolution.get("selected_dataset_ids"), list) else trace_metadata.get("dataset_ids")
            if dataset_ids:
                plan_payload.setdefault("dataset_ids", dataset_ids)
                plan_payload.setdefault("dataset_id", dataset_ids[0] if len(dataset_ids) == 1 else "")
            plan_payload.setdefault("dataset_scope", dataset_resolution.get("scope") or trace_metadata.get("dataset_scope") or "")
            workspace = BranchWorkspaceManager.from_payload(metadata.get("branch_workspace"))
            workspace = upsert_branch_from_plan(
                workspace,
                plan_payload,
                artifact_count=len(getattr(investigation, "artifacts", []) or []),
                finding_count=len(getattr(investigation, "findings", []) or []),
            )
            metadata["branch_workspace"] = workspace.to_payload()
            metadata["conversation_state"]["branch_workspace"] = workspace.to_payload()
            metadata["conversation_state"]["active_branch_id"] = workspace.active_branch_id
            if plan_payload.get("metric"):
                metadata["conversation_state"]["active_metric"] = plan_payload.get("metric")
            if plan_payload.get("dimension"):
                metadata["conversation_state"]["active_dimension"] = plan_payload.get("dimension")
            if plan_payload.get("time_axis"):
                metadata["conversation_state"]["active_time_axis"] = plan_payload.get("time_axis")
            if plan_payload.get("chart_type"):
                metadata["conversation_state"]["active_chart_type"] = plan_payload.get("chart_type")
            if plan_payload.get("aggregation"):
                metadata["conversation_state"]["active_aggregation"] = plan_payload.get("aggregation")
            if plan_payload.get("filters"):
                metadata["conversation_state"]["active_filters"] = plan_payload.get("filters")
            if plan_payload.get("dataset_scope"):
                metadata["conversation_state"]["active_dataset_scope"] = plan_payload.get("dataset_scope")
            if plan_payload.get("dataset_id"):
                metadata["conversation_state"]["active_dataset_id"] = plan_payload.get("dataset_id")
            if plan_payload.get("dataset_ids"):
                metadata["conversation_state"]["active_dataset_ids"] = plan_payload.get("dataset_ids")
        elif routing_decision.branch_action.value in {"global", "create"} and routing_decision.question_intent_type.value not in {"direct_analysis", "distribution_analysis", "transformation", "artifact_explanation"}:
            dataset_resolution = trace_metadata.get("dataset_resolution") if isinstance(trace_metadata.get("dataset_resolution"), dict) else {}
            dataset_ids = dataset_resolution.get("selected_dataset_ids") if isinstance(dataset_resolution.get("selected_dataset_ids"), list) else []
            workspace = BranchWorkspaceManager.from_payload(metadata.get("branch_workspace"))
            workspace = upsert_branch_for_intent(
                workspace,
                intent_type=routing_decision.question_intent_type.value,
                dataset_scope=str(dataset_resolution.get("scope") or ""),
                dataset_id=str(dataset_ids[0]) if len(dataset_ids) == 1 else "",
                dataset_ids=dataset_ids,
                created_from_query=question,
                parent_branch_id=previous.get("active_branch_id") if isinstance(previous, dict) else None,
            )
            metadata["branch_workspace"] = workspace.to_payload()
            metadata["conversation_state"]["branch_workspace"] = workspace.to_payload()
            metadata["conversation_state"]["active_branch_id"] = workspace.active_branch_id
        latest_transformation = latest_chart.get("active_transformation") if isinstance(latest_chart, dict) else ""
        if latest_transformation:
            metadata["conversation_state"]["active_transformation"] = latest_transformation
            metadata["conversation_state"]["previous_transformation"] = previous.get("active_transformation") if isinstance(previous, dict) else ""
            metadata["conversation_state"]["active_adjusted_ranking"] = latest_chart.get("active_adjusted_ranking") or []
            rich_transformation = latest_chart.get("active_transformation_result") if isinstance(latest_chart.get("active_transformation_result"), dict) else {}
            metadata["conversation_state"]["active_transformation_result"] = rich_transformation or {
                    "transformation_type": latest_transformation,
                    "metric": latest_chart.get("metric") or "",
                    "dimension": latest_chart.get("dimension") or "",
                    "filtered_ranking": latest_chart.get("active_adjusted_ranking") or [],
                    "ranking_scope": latest_chart.get("ranking_scope") or "",
                    "row_count": latest_chart.get("row_count"),
                    "interpretation_summary": report_summary,
                }
            metadata["conversation_state"]["last_successful_analysis_type"] = latest_transformation
            metadata["conversation_state"]["active_aggregation"] = latest_chart.get("aggregation") or ""
            metadata["conversation_state"]["active_ranking_scope"] = latest_chart.get("ranking_scope") or ""
            metadata["conversation_state"]["transformation_lineage"] = {
                "base_artifact_id": latest_chart.get("parent_artifact_id") or latest_chart.get("base_artifact_id") or "",
                "base_dimension": latest_chart.get("base_dimension") or latest_chart.get("dimension") or "",
                "base_metric": latest_chart.get("base_metric") or latest_chart.get("metric") or "",
                "base_aggregation": latest_chart.get("base_aggregation") or latest_chart.get("aggregation") or "",
                "transformation_type": latest_transformation,
                "transformation_params": rich_transformation.get("threshold") if isinstance(rich_transformation, dict) else {},
                "derived_artifact_id": latest_chart.get("derived_artifact_id") or latest_chart.get("artifact_id") or "",
            }
        updater = getattr(store, "update_investigation_metadata", None)
        if callable(updater):
            updater(investigation_id, metadata)
        else:
            investigation.metadata = metadata
    except Exception:
        return


def _compact_text(value: str | None, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _multi_dataset_lifecycle_trace(
    *,
    question: str,
    dataset_registry: list[Any],
    dataset_resolution: Any,
    routing_decision: Any,
    trace_metadata: dict[str, Any],
    artifacts: list[Any],
    findings: list[Any],
    run_id: str,
) -> list[dict[str, Any]]:
    scopes = trace_metadata.get("dataset_execution_scopes") if isinstance(trace_metadata.get("dataset_execution_scopes"), list) else []
    run_artifacts = [artifact for artifact in artifacts if getattr(artifact, "run_id", None) == run_id]
    run_findings = [finding for finding in findings if getattr(finding, "run_id", None) == run_id]
    return [
        {"stage": "question", "question": question},
        {"stage": "dataset_registry", "dataset_ids": [entry.dataset_id for entry in dataset_registry], "count": len(dataset_registry)},
        {"stage": "dataset_semantic_profiles", "profiles": [{"dataset_id": entry.dataset_id, "roles": entry.semantic_profile.get("roles", [])[:8], "concepts": entry.semantic_profile.get("concepts", [])[:8]} for entry in dataset_registry]},
        {"stage": "compatibility_scoring", "scores": {scope.get("dataset_id"): scope.get("compatibility_score") for scope in scopes if isinstance(scope, dict)}},
        {"stage": "operation_classification", "operation": trace_metadata.get("operation"), "intent": getattr(getattr(routing_decision, "question_intent_type", None), "value", "")},
        {"stage": "planner", "scope": getattr(dataset_resolution, "scope", ""), "selected_dataset_ids": list(getattr(dataset_resolution, "selected_dataset_ids", []) or [])},
        {"stage": "dataset_routing", "branch_count": len(scopes), "branch_dataset_ids": [scope.get("dataset_id") for scope in scopes if isinstance(scope, dict)]},
        {"stage": "role_assignment", "roles_by_dataset": {scope.get("dataset_id"): scope.get("semantic_roles") for scope in scopes if isinstance(scope, dict)}},
        {"stage": "execution", "computed_result_counts": {scope.get("dataset_id"): len(scope.get("computed_results") or []) for scope in scopes if isinstance(scope, dict)}},
        {"stage": "evidence_generation", "finding_counts": {scope.get("dataset_id"): len(scope.get("findings") or []) for scope in scopes if isinstance(scope, dict)}},
        {"stage": "artifacts", "artifact_count": len(run_artifacts), "artifact_dataset_ids": [getattr(artifact, "metadata", {}).get("dataset_id") for artifact in run_artifacts]},
        {"stage": "findings", "finding_count": len(run_findings)},
        {"stage": "synthesis", "analysis_type": trace_metadata.get("analysis_type"), "dataset_ids": trace_metadata.get("dataset_ids") or []},
        {"stage": "final_response", "branch_count": trace_metadata.get("branch_count"), "operation": trace_metadata.get("operation")},
    ]


def _unique_ids(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value and value not in seen:
            unique.append(value)
            seen.add(value)
    return unique


def _normalize_analysis_mode(value: str | None) -> str:
    normalized = str(value or "exploration").strip().lower().replace(" ", "_")
    return normalized if normalized in {"exploration", "validation", "executive", "data_quality"} else "exploration"


def _stage_label(stage: InvestigationRunStage) -> str:
    labels = {
        InvestigationRunStage.PREPARING_DATA: "Preparing data",
        InvestigationRunStage.BUILDING_CONTEXT: "Building context",
        InvestigationRunStage.RUNNING_ANALYSIS: "Running analysis",
        InvestigationRunStage.VALIDATING_RESULTS: "Validating results",
        InvestigationRunStage.GENERATING_REPORT: "Generating report",
        InvestigationRunStage.COMPLETED: "Latest pass",
    }
    return labels.get(stage, stage.value.replace("_", " ").title())


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
