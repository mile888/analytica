from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

from source.product.adapter import agent_output_to_investigation_update
from source.product.conversation_engine import answer_from_conversation_state, clarification_from_state, response_quality_gate
from source.product.data_context import build_data_source_usage_context, usage_context_to_prompt
from source.product.evidence_resolution import is_evidence_followup
from source.product.execution_context import (
    ExecutionContextUnavailableError,
    execution_context_failure_output,
    execution_required_for_question,
    execution_unavailable_in_context,
)
from source.product.fallback_analysis import deterministic_context_fallback, deterministic_investigation_fallback
from source.product.investigation import (
    Artifact,
    ArtifactType,
    ArtifactVisibility,
    Investigation,
    InvestigationRun,
    InvestigationStatus,
    new_id,
    utc_now,
)
from source.product.store import InvestigationStore


Runner = Callable[..., dict[str, Any]]


def default_agent_runner(*, question: str, df: Any = None, data_context: dict[str, Any] | None = None) -> dict[str, Any]:
    from source.agent import run_once

    context = data_context or {}
    if df is None:
        if execution_required_for_question(question, data_context=context) and execution_unavailable_in_context(context):
            error = _execution_error_from_context(context)
            return execution_context_failure_output(question, error)
        context_fallback = deterministic_context_fallback(question, context)
        if context_fallback:
            return context_fallback
    question_for_agent = _question_with_data_context(question, context)
    return run_once(
        df,
        question_for_agent,
        messages=context.get("messages"),
        engine=context.get("engine", "auto"),
        thread_id=context.get("thread_id"),
    )


class InvestigationService:
    def __init__(self, store: InvestigationStore | None = None, runner: Runner | None = None) -> None:
        self.store = store or InvestigationStore()
        self.runner = runner or default_agent_runner

    def create_investigation(self, question: str, title: Optional[str] = None) -> Investigation:
        return self.store.create_investigation(question=question, title=title)

    def run_investigation(
        self,
        investigation_id: str,
        df: Any = None,
        data_context: dict[str, Any] | None = None,
    ) -> Investigation:
        investigation = self.store.get_investigation(investigation_id)
        self.store.update_status(investigation_id, InvestigationStatus.RUNNING)
        initial_context = dict(data_context or {})
        run = InvestigationRun(
            status=InvestigationStatus.RUNNING,
            investigation_id=investigation_id,
            run_id=initial_context.get("product_run_id") or new_id("run"),
        )

        try:
            run_context = dict(initial_context)
            data_source_ids = _unique_ids(
                list(run_context.get("data_source_ids", [])) + list(investigation.linked_data_source_ids)
            )
            usage_contexts = []
            for data_source_id in data_source_ids:
                try:
                    self.store.link_data_source_to_investigation(investigation_id, data_source_id)
                    usage_contexts.append(build_data_source_usage_context(self.store, data_source_id))
                except KeyError:
                    continue
            if usage_contexts:
                run_context["data_source_ids"] = data_source_ids
                run_context["data_source_usage_contexts"] = [_jsonable(context) for context in usage_contexts]
                run_context["data_context_prompt"] = usage_context_to_prompt(usage_contexts)
            active_question = str(run_context.get("active_question") or investigation.user_question)
            execution_required = execution_required_for_question(active_question, df=df, data_context=run_context)
            execution_unavailable = execution_unavailable_in_context(run_context)
            if execution_required and df is None and execution_unavailable:
                run.output = execution_context_failure_output(active_question, _execution_error_from_context(run_context))
                update = agent_output_to_investigation_update(run.output)
                run.trace = update.trace
                run.status = InvestigationStatus.NEEDS_REVIEW
                run.finished_at = utc_now()
                self.store.add_run(investigation_id, run)
                if update.report:
                    update.report.run_id = run.run_id
                    update.report.question = update.report.question or active_question
                    self.store.set_report(investigation_id, update.report)
                self.store.update_status(investigation_id, InvestigationStatus.NEEDS_REVIEW)
                return self.store.get_investigation(investigation_id)
            output = self.runner(
                question=active_question,
                df=df,
                data_context=run_context,
            )
            run.output = output if isinstance(output, dict) else {"raw_output": output}
            if not is_evidence_followup(active_question):
                engine_answer = answer_from_conversation_state(
                    question=active_question,
                    conversation_context=run_context.get("conversation_context") if isinstance(run_context, dict) else {},
                    recent_artifacts=(run_context.get("conversation_context") or {}).get("recent_artifacts", []) if isinstance(run_context.get("conversation_context"), dict) else [],
                )
                if engine_answer:
                    run.output = _conversation_response_output(active_question, engine_answer)
            if self._has_runner_error(run.output):
                if execution_required and df is None and execution_unavailable:
                    run.output = execution_context_failure_output(active_question, _execution_error_from_context(run_context))
                    update = agent_output_to_investigation_update(run.output)
                    raise _ControlledExecutionContextFailure(update)
                fallback = deterministic_investigation_fallback(active_question, df, data_context=run_context)
                if not fallback:
                    fallback = deterministic_context_fallback(active_question, run_context)
                if fallback:
                    fallback["fallback_from"] = run.output
                    run.output = fallback
                else:
                    raise RuntimeError(str(run.output.get("exec_error") or "Investigation runner failed."))
            if _should_prefer_deterministic_answer(active_question, df):
                deterministic = deterministic_investigation_fallback(active_question, df, data_context=run_context)
                if deterministic:
                    deterministic["fallback_from"] = run.output
                    run.output = deterministic
            if not is_useful_analytical_answer(_answer_text_from_output(run.output)):
                if execution_required and df is None and execution_unavailable:
                    run.output = execution_context_failure_output(active_question, _execution_error_from_context(run_context))
                    update = agent_output_to_investigation_update(run.output)
                    raise _ControlledExecutionContextFailure(update)
                fallback = deterministic_investigation_fallback(active_question, df, data_context=run_context)
                if not fallback:
                    fallback = deterministic_context_fallback(active_question, run_context)
                if fallback:
                    fallback["fallback_from"] = run.output
                    run.output = fallback
                else:
                    run.output = _safe_no_answer_output(active_question, run_context)
            update = agent_output_to_investigation_update(run.output)
            if _should_replace_generic_overview(active_question, update, df):
                analytical_fallback = deterministic_investigation_fallback(active_question, df, data_context=run_context)
                if analytical_fallback:
                    analytical_fallback["fallback_from"] = run.output
                    run.output = analytical_fallback
                    update = agent_output_to_investigation_update(run.output)
            valid, _quality_reason = response_quality_gate(
                question=active_question,
                response_text=_answer_text_from_output(run.output),
                conversation_context=run_context.get("conversation_context") if isinstance(run_context, dict) else {},
            )
            if not valid:
                engine_answer = None
                if _is_transformation_impact_followup(active_question):
                    engine_answer = answer_from_conversation_state(
                        question=active_question,
                        conversation_context=run_context.get("conversation_context") if isinstance(run_context, dict) else {},
                        recent_artifacts=(run_context.get("conversation_context") or {}).get("recent_artifacts", []) if isinstance(run_context.get("conversation_context"), dict) else [],
                    )
                analytical_fallback = None if engine_answer else deterministic_investigation_fallback(active_question, df, data_context=run_context)
                if analytical_fallback:
                    analytical_fallback["fallback_from"] = run.output
                    run.output = analytical_fallback
                    update = agent_output_to_investigation_update(run.output)
                else:
                    engine_answer = engine_answer or answer_from_conversation_state(
                            question=active_question,
                            conversation_context=run_context.get("conversation_context") if isinstance(run_context, dict) else {},
                            recent_artifacts=(run_context.get("conversation_context") or {}).get("recent_artifacts", []) if isinstance(run_context.get("conversation_context"), dict) else [],
                        )
                    if engine_answer:
                        run.output = _conversation_response_output(active_question, engine_answer)
                        update = agent_output_to_investigation_update(run.output)
                fallback_valid, _ = response_quality_gate(
                    question=active_question,
                    response_text=_answer_text_from_output(run.output),
                    conversation_context=run_context.get("conversation_context") if isinstance(run_context, dict) else {},
                )
                if not fallback_valid:
                    clarification = clarification_from_state(
                        run_context.get("conversation_context") if isinstance(run_context, dict) else {}
                    )
                    if clarification:
                        run.output = _conversation_response_output(active_question, clarification)
                        update = agent_output_to_investigation_update(run.output)
            if _is_chart_request(active_question) and not _update_has_chart(update):
                if execution_required and df is None and execution_unavailable:
                    run.output = execution_context_failure_output(active_question, _execution_error_from_context(run_context))
                    update = agent_output_to_investigation_update(run.output)
                    raise _ControlledExecutionContextFailure(update)
                chart_fallback = deterministic_investigation_fallback(active_question, df, data_context=run_context)
                if chart_fallback:
                    chart_fallback["fallback_from"] = run.output
                    run.output = chart_fallback
                    update = agent_output_to_investigation_update(run.output)
            run.trace = update.trace
            run.status = InvestigationStatus.NEEDS_REVIEW
            run.finished_at = utc_now()

            self.store.add_run(investigation_id, run)
            trace_metadata = run.output.get("trace_metadata") if isinstance(run.output, dict) and isinstance(run.output.get("trace_metadata"), dict) else {}
            artifact_ids: list[str] = []
            previous_chart_artifact_id = _latest_chart_artifact_id(getattr(investigation, "artifacts", []) or [])
            for artifact in update.artifacts:
                artifact.run_id = run.run_id
                artifact.metadata = _enrich_artifact_metadata(
                    artifact,
                    trace_metadata=trace_metadata,
                    message_id=(run_context or {}).get("active_message_id") if isinstance(run_context, dict) else None,
                )
                if artifact.metadata.get("transformation_type") and previous_chart_artifact_id:
                    if not artifact.metadata.get("parent_artifact_id"):
                        artifact.metadata["parent_artifact_id"] = previous_chart_artifact_id
                    if not artifact.metadata.get("base_artifact_id"):
                        artifact.metadata["base_artifact_id"] = previous_chart_artifact_id
                self.store.add_artifact(investigation_id, artifact)
                if artifact.visibility == ArtifactVisibility.USER:
                    artifact_ids.append(artifact.artifact_id)
            for finding in update.findings:
                finding.run_id = run.run_id
                finding.evidence_artifact_ids = list(artifact_ids)
                finding.metadata.setdefault("supporting_artifact_ids", list(artifact_ids))
                finding.metadata["linked_output_count"] = len(artifact_ids)
                self.store.add_finding(investigation_id, finding)
            if update.report:
                update.report.run_id = run.run_id
                update.report.question = update.report.question or active_question
                update.report.artifact_ids = list(artifact_ids)
                self.store.set_report(investigation_id, update.report)
            self.store.update_status(investigation_id, InvestigationStatus.NEEDS_REVIEW)
        except _ControlledExecutionContextFailure as exc:
            run.trace = exc.update.trace
            run.status = InvestigationStatus.NEEDS_REVIEW
            run.finished_at = utc_now()
            self.store.add_run(investigation_id, run)
            if exc.update.report:
                exc.update.report.run_id = run.run_id
                exc.update.report.question = exc.update.report.question or active_question
                self.store.set_report(investigation_id, exc.update.report)
            self.store.update_status(investigation_id, InvestigationStatus.NEEDS_REVIEW)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            run.status = InvestigationStatus.FAILED
            run.error = message
            run.finished_at = utc_now()
            self.store.add_run(investigation_id, run)
            self.store.add_artifact(
                investigation_id,
                Artifact(
                    artifact_type=ArtifactType.VALIDATION,
                    title="Run error",
                    content=message,
                    visibility=ArtifactVisibility.TECHNICAL,
                    run_id=run.run_id,
                    metadata={"kind": "error"},
                ),
            )
            self.store.update_status(investigation_id, InvestigationStatus.FAILED)

        return self.store.get_investigation(investigation_id)

    @staticmethod
    def _has_runner_error(output: dict[str, Any]) -> bool:
        return bool(output.get("exec_error")) or str(output.get("critic_verdict", "")).upper() == "ERROR"


class _ControlledExecutionContextFailure(Exception):
    def __init__(self, update: Any) -> None:
        super().__init__("Execution context unavailable")
        self.update = update


def _execution_error_from_context(context: dict[str, Any]) -> ExecutionContextUnavailableError:
    failures = context.get("execution_context_failures") if isinstance(context, dict) else None
    if isinstance(failures, list) and failures:
        first = failures[0] if isinstance(failures[0], dict) else {}
        return ExecutionContextUnavailableError(
            str(first.get("message") or "Executable dataset rows are unavailable."),
            data_source_id=first.get("data_source_id"),
            reason=str(first.get("reason") or "raw_rows_unavailable"),
        )
    return ExecutionContextUnavailableError(
        "Executable dataset rows are unavailable. The saved schema/profile can be used for metadata reasoning only.",
        reason="raw_rows_unavailable",
    )


def _question_with_data_context(question: str, data_context: dict[str, Any]) -> str:
    sections = [
        (
            "Analytical continuity rules:\n"
            "- Answer the CURRENT user question first.\n"
            "- Treat this as an ongoing analytical investigation, but never explain that mechanism to the user.\n"
            "- Use prior findings, artifacts, reports, and notes only when they help answer the current question.\n"
            "- Do not repeat the dataset overview or column inventory unless the user explicitly asks for it.\n"
            "- Resolve references like 'this', 'that', 'here', or 'the chart' from the recent investigation context.\n"
            "- Prefer new reasoning, interpretation, evidence strength, and next analytical steps.\n"
            "- Never mention orchestration, memory, context handling, run lifecycle, backend workflow, or prompt mechanics.\n"
            "- Sound like a senior data analyst: give comparisons, rankings, anomalies, caveats, charts, and recommendations.\n"
            "- When useful, structure important insights with: insight, confidence, evidence, limitation, recommended action.\n"
            "- Do analytics; do not talk about doing analytics."
        )
    ]
    context_prompt = data_context.get("data_context_prompt")
    if context_prompt:
        sections.append("Use this data source context when planning the analysis:\n" + str(context_prompt))
    conversation_prompt = _conversation_context_prompt(data_context.get("conversation_context"))
    if conversation_prompt:
        sections.append("Use this investigation context for continuity:\n" + conversation_prompt)
    mode_prompt = _analysis_mode_prompt(data_context.get("analysis_mode"))
    if mode_prompt:
        sections.append(mode_prompt)
    if not sections:
        return question
    return question + "\n\n" + "\n\n".join(sections)


def _analysis_mode_prompt(value: Any) -> str:
    mode = str(value or "exploration").strip().lower()
    prompts = {
        "exploration": (
            "Investigation mode: Exploration. Prioritize useful patterns, surprising segments, anomalies, and chartable comparisons."
        ),
        "validation": (
            "Investigation mode: Validation. Prioritize evidence strength, confidence, limitations, counterexamples, and what would disprove the conclusion."
        ),
        "executive": (
            "Investigation mode: Executive. Prioritize business impact, concise conclusions, decision implications, and report-ready recommendations."
        ),
        "data_quality": (
            "Investigation mode: Data quality. Prioritize missing values, duplicates, inconsistent values, outliers, and preprocessing recommendations."
        ),
    }
    return prompts.get(mode, prompts["exploration"])


def _conversation_context_prompt(context: Any) -> str:
    if not isinstance(context, dict):
        return ""
    lines: list[str] = []
    initial_question = str(context.get("initial_question") or "").strip()
    if initial_question:
        lines.append(f"Original analytical question: {initial_question}")
    resolved_intent = context.get("resolved_intent")
    if isinstance(resolved_intent, dict):
        lines.append(
            "Current user intent: "
            + str(resolved_intent.get("primary") or "")
            + ("; components: " + ", ".join(str(item) for item in resolved_intent.get("components", [])[:5]) if resolved_intent.get("components") else "")
        )
    conversation_state = context.get("conversation_state")
    if isinstance(conversation_state, dict):
        state_parts = []
        for key in ("phase", "active_topic", "active_metric", "active_dimension", "active_hypothesis", "current_objective"):
            value = str(conversation_state.get(key) or "").strip()
            if value:
                state_parts.append(f"{key}={value}")
        if state_parts:
            lines.append("Persistent investigation state: " + "; ".join(state_parts))
    messages = context.get("messages")
    if isinstance(messages, list) and messages:
        lines.append("Recent conversation:")
        for message in messages[-8:]:
            if not isinstance(message, dict):
                continue
            role = "Analyst" if str(message.get("role") or "") == "user" else "AI analyst"
            content = str(message.get("content") or "").strip()
            if content:
                lines.append(f"- {role}: {content}")
    findings = context.get("latest_findings")
    if isinstance(findings, list) and findings:
        lines.append("Current insights:")
        lines.extend(f"- {item}" for item in findings[:5] if item)
    thread_state = context.get("thread_state")
    if isinstance(thread_state, dict):
        metric = str(thread_state.get("active_metric") or "").strip()
        dimension = str(thread_state.get("active_dimension") or "").strip()
        time_axis = str(thread_state.get("active_time_axis") or "").strip()
        chart_type = str(thread_state.get("active_chart_type") or "").strip()
        thread_parts = [
            f"metric={metric}" if metric else "",
            f"dimension={dimension}" if dimension else "",
            f"time_axis={time_axis}" if time_axis else "",
            f"chart_type={chart_type}" if chart_type else "",
        ]
        thread_line = ", ".join(part for part in thread_parts if part)
        if thread_line:
            lines.append("Active analytical topic: " + thread_line)
    report_summary = str(context.get("latest_report_summary") or "").strip()
    if report_summary:
        lines.append(f"Current analytical narrative: {report_summary}")
    artifact_titles = context.get("artifact_titles")
    if isinstance(artifact_titles, list) and artifact_titles:
        lines.append("Available outputs: " + ", ".join(str(item) for item in artifact_titles[:8] if item))
    return "\n".join(lines)


def _unique_ids(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value and value not in seen:
            unique.append(value)
            seen.add(value)
    return unique


def _is_chart_request(question: str) -> bool:
    text = " ".join(str(question or "").lower().replace("_", " ").split())
    markers = (
        "chart",
        "plot",
        "graph",
        "visual",
        "histogram",
        "build a chart",
        "show top",
        "top ",
        " by ",
        "график",
        "диаграм",
        "визуал",
        "построй",
        "построить",
        "нарисуй",
    )
    return any(marker in text for marker in markers)


def _is_overview_request(question: str) -> bool:
    text = " ".join(str(question or "").lower().replace("_", " ").split())
    markers = (
        "what can you say",
        "summarize",
        "overview",
        "describe dataset",
        "describe data",
        "какие данные",
        "какие тут данные",
        "что можешь сказать",
        "что ты можешь сказать",
        "опиши данные",
        "опиши датасет",
    )
    return any(marker in text for marker in markers)


def _is_dependency_request(question: str) -> bool:
    text = " ".join(str(question or "").lower().replace("_", " ").split())
    markers = (
        "dependency",
        "dependencies",
        "relationship",
        "relationships",
        "related",
        "affect",
        "affects",
        "drivers",
        "what drives",
        "завис",
        "связ",
        "влияет",
        "коррел",
    )
    return any(marker in text for marker in markers)


def _is_quality_request(question: str) -> bool:
    text = " ".join(str(question or "").lower().replace("_", " ").split())
    markers = (
        "quality",
        "reliable",
        "reliability",
        "missing",
        "duplicates",
        "duplicate",
        "clean",
        "preprocess",
        "null",
        "пропуск",
        "дублик",
        "качеств",
    )
    return any(marker in text for marker in markers)


def _should_prefer_deterministic_answer(question: str, df: Any) -> bool:
    if df is None:
        return False
    text = " ".join(str(question or "").lower().replace("_", " ").split())
    return (
        (_is_overview_request(question) and _is_dependency_request(question))
        or _is_quality_request(question)
        or _is_chart_request(question)
        or any(marker in text for marker in ("top ", "highest", "lowest", "rank", "ranking", "average", "mean", "median", " by ", "distribution", "histogram", "spread", "trend", "over time", "time series", "growth", "season", "seasonality", "shipping", "ship ", "delivery", "anomalous period", "anomal", "outlier", "extreme", "without", "excluding", "exclude", "remove", "sparse", "normalize", "unusual", "leader", "leaders", "remain", "hypothesis", "validate", "formula", "dtype", "dtypes", "data type", "convert numeric", "create ", "add column", "derive", "delimiter", "separator", "test this", "impact", "affect", "динамик", "во времени", "по времени", "рост", "сезон", "достав", "аномальн период", "выброс", "экстрем", "убрать", "исключ", "без ", "медиан", "нормализ", "маленькие выборки", "аномал", "лидер", "остаются", "остались", "гипотез", "проверить", "сильн", "strongest", "volume", "объем", "объём", "колич", "заказ", "вли", "compare", "сравн", "теперь", " по ", "бизнес вопрос", "можно исследовать", "важные поля", "важны", "important fields", "business question"))
    )


def _should_replace_generic_overview(question: str, update: Any, df: Any) -> bool:
    if df is None:
        return False
    if _is_overview_request(question) and not _is_dependency_request(question):
        return False
    if _is_chart_request(question):
        return False
    text = " ".join(
        str(value or "")
        for value in [
            getattr(getattr(update, "report", None), "summary", ""),
            getattr(getattr(update, "report", None), "answer", ""),
            " ".join(getattr(finding, "text", "") for finding in getattr(update, "findings", []) or []),
        ]
    ).lower()
    overview_markers = (
        "rows and",
        "columns",
        "quantitative fields",
        "segmentation",
        "dataset has",
        "основу для аналитического расследования",
        "строк и",
        "колонок",
        "количественные точки входа",
        "для сегментации",
    )
    analytical_markers = (
        "highest",
        "lowest",
        "average",
        "median",
        "correlation",
        "outlier",
        "differs",
        "varies",
        "chart",
        "график",
        "выше",
        "ниже",
        "средн",
        "выброс",
        "отлич",
        "trend",
        "growth",
        "season",
        "period",
        "time",
        "динамик",
        "рост",
        "период",
        "сезон",
    )
    return any(marker in text for marker in overview_markers) and not any(marker in text for marker in analytical_markers)


def is_useful_analytical_answer(text: str) -> bool:
    cleaned = " ".join(str(text or "").split())
    normalized = cleaned.lower()
    if not cleaned:
        return False
    forbidden_exact = {"done", "ok", "complete", "completed"}
    if normalized in forbidden_exact:
        return False
    forbidden_fragments = (
        "analysis completed",
        "run completed",
        "finished the analysis",
        "available above",
        "keep exploring",
        "investigation updated",
        "investigation has been updated",
        "new analytical material",
        "analysis pass updated",
        "continue with another follow-up",
    )
    if any(fragment in normalized for fragment in forbidden_fragments):
        analytical_markers = (
            "highest",
            "lowest",
            "average",
            "median",
            "correlation",
            "outlier",
            "differs",
            "varies",
            "concentrated",
            "variance",
            "spread",
            "chart",
            "evidence",
            "выше",
            "ниже",
            "средн",
            "выброс",
            "отлич",
            "концентр",
            "разброс",
        )
        return any(marker in normalized for marker in analytical_markers)
    return True


def _answer_text_from_output(output: dict[str, Any]) -> str:
    structured = output.get("structured_report") if isinstance(output.get("structured_report"), dict) else {}
    values = [
        output.get("summary"),
        output.get("final_answer"),
        structured.get("summary"),
        " ".join(str(item) for item in output.get("key_findings", []) if str(item).strip()) if isinstance(output.get("key_findings"), list) else "",
        " ".join(str(item) for item in structured.get("key_findings", []) if str(item).strip()) if isinstance(structured.get("key_findings"), list) else "",
    ]
    return " ".join(str(value or "").strip() for value in values if str(value or "").strip())


def _conversation_response_output(question: str, response: Any) -> dict[str, Any]:
    text = str(getattr(response, "text", "") or "")
    return {
        "summary": text,
        "final_answer": text,
        "structured_report": {
            "question": question,
            "summary": text,
            "key_findings": list(getattr(response, "findings", []) or []),
            "evidence": ["Generated by conversation engine from active investigation state."],
            "limitations": [],
            "next_steps": [],
            "tool_timeline": [{"tool": "conversation_engine", "status": "ok", "response_kind": getattr(response, "response_kind", "")}],
        },
        "tool_timeline": [{"tool": "conversation_engine", "status": "ok", "response_kind": getattr(response, "response_kind", "")}],
        "trace_metadata": {"fallback": "conversation_engine", "analysis_type": getattr(response, "response_kind", "")},
        "artifacts": list(getattr(response, "artifacts", []) or []),
        "critic_verdict": "",
    }


def _safe_no_answer_output(question: str, data_context: dict[str, Any]) -> dict[str, Any]:
    context = data_context.get("conversation_context") if isinstance(data_context, dict) else {}
    chart = context.get("latest_chart_context") if isinstance(context, dict) and isinstance(context.get("latest_chart_context"), dict) else {}
    metric = str(chart.get("metric") or "").strip()
    dimension = str(chart.get("dimension") or "").strip()
    if metric and dimension:
        summary = (
            f"I could not compute a fresh result for this question. The current analytical thread is `{metric}` by `{dimension}`; "
            "the next reliable check is to compare the strongest groups, inspect anomalies, or test whether volume explains the gap."
        )
    else:
        summary = (
            "I could not compute a fresh result for this question. Ask a focused question about a specific metric, segment, trend, anomaly, chart, "
            "or evidence gap so the next answer can be grounded in the dataset."
        )
    return {
        "summary": summary,
        "key_findings": [],
        "limitations": ["A fresh analytical calculation was not available for this response."],
        "next_steps": ["Ask a focused analytical follow-up tied to a metric, group, trend, anomaly, chart, or evidence gap."],
        "trace_metadata": {"fallback": "safe_visible_assistant_response", "suppress_key_findings": True, "analysis_type": "fallback"},
    }


def _is_transformation_impact_followup(question: str) -> bool:
    text = " ".join(str(question or "").lower().split())
    return any(marker in text for marker in ("what changed", "changed after", "after filtering", "became unreliable", "что измен"))


def _update_has_chart(update: Any) -> bool:
    return any(getattr(artifact, "artifact_type", None) == ArtifactType.CHART for artifact in getattr(update, "artifacts", []) or [])


def _output_has_chart(output: dict[str, Any]) -> bool:
    artifacts = output.get("artifacts")
    if not isinstance(artifacts, list):
        return False
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        if str(artifact.get("artifact_type") or artifact.get("type") or "").lower() == "chart":
            return True
    return False


def _enrich_artifact_metadata(artifact: Artifact, *, trace_metadata: dict[str, Any], message_id: str | None) -> dict[str, Any]:
    metadata = dict(getattr(artifact, "metadata", {}) or {})
    nested_metadata = metadata.get("metadata") if isinstance(metadata.get("metadata"), dict) else {}
    content = getattr(artifact, "content", None)
    content_dict = content if isinstance(content, dict) else {}
    query_plan = metadata.get("query_plan") if isinstance(metadata.get("query_plan"), dict) else trace_metadata.get("query_plan") if isinstance(trace_metadata.get("query_plan"), dict) else {}
    branch_route = trace_metadata.get("branch_route") if isinstance(trace_metadata.get("branch_route"), dict) else {}
    metadata.setdefault("artifact_id", artifact.artifact_id)
    metadata.setdefault("branch_id", metadata.get("branch_id") or nested_metadata.get("branch_id") or trace_metadata.get("branch_id") or branch_route.get("branch_id") or "global")
    metadata.setdefault("branch_title", metadata.get("branch_title") or _artifact_branch_title(metadata, content_dict, query_plan))
    metadata.setdefault("metric", metadata.get("metric") or nested_metadata.get("metric") or content_dict.get("metric") or query_plan.get("metric"))
    metadata.setdefault("dimension", metadata.get("dimension") or nested_metadata.get("dimension") or content_dict.get("dimension") or content_dict.get("x") or query_plan.get("dimension"))
    metadata.setdefault("aggregation", metadata.get("aggregation") or nested_metadata.get("aggregation") or content_dict.get("aggregation") or query_plan.get("aggregation"))
    metadata.setdefault("filters", metadata.get("filters") or nested_metadata.get("filters") or content_dict.get("filters") or query_plan.get("filters") or [])
    metadata.setdefault("chart_type", metadata.get("chart_type") or nested_metadata.get("chart_type") or content_dict.get("chart_type"))
    metadata.setdefault("transformation_state", metadata.get("transformation_state") or nested_metadata.get("transformation_type") or nested_metadata.get("analysis_type") or query_plan.get("transformation") or trace_metadata.get("analysis_type") or "raw")
    for key in (
        "base_metric",
        "base_dimension",
        "base_aggregation",
        "transformation_type",
        "transformation_params",
        "parent_artifact_id",
        "base_artifact_id",
    ):
        if key not in metadata and key in nested_metadata:
            metadata[key] = nested_metadata.get(key)
        if key not in metadata and key in content_dict:
            metadata[key] = content_dict.get(key)
    metadata.setdefault("created_from_message_id", message_id or "")
    metadata.setdefault("created_from_query", query_plan.get("raw_question") or "")
    metadata.setdefault("created_at", artifact.created_at.isoformat())
    return metadata


def _latest_chart_artifact_id(artifacts: list[Any]) -> str:
    for artifact in reversed(list(artifacts or [])):
        artifact_type = getattr(getattr(artifact, "artifact_type", None), "value", getattr(artifact, "artifact_type", None))
        if str(artifact_type or "").lower() == "chart":
            return str(getattr(artifact, "artifact_id", "") or getattr(artifact, "id", "") or "")
    return ""


def _artifact_branch_title(metadata: dict[str, Any], content: dict[str, Any], query_plan: dict[str, Any]) -> str:
    metric = str(metadata.get("metric") or content.get("metric") or query_plan.get("metric") or "").strip()
    dimension = str(metadata.get("dimension") or content.get("dimension") or content.get("x") or query_plan.get("dimension") or "").strip()
    time_axis = str(content.get("time_axis") or content.get("timestamp") or query_plan.get("time_axis") or "").strip()
    chart_type = str(metadata.get("chart_type") or content.get("chart_type") or query_plan.get("chart_type") or "").strip()
    if metric and time_axis:
        return f"{metric} Trend over Time"
    if metric and chart_type == "histogram":
        return f"{metric} Distribution"
    if metric and dimension:
        return f"{metric} by {dimension}"
    return "Global output"


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
