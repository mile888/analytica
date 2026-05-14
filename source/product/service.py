from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

from source.product.adapter import agent_output_to_investigation_update
from source.product.data_context import build_data_source_usage_context, usage_context_to_prompt
from source.product.fallback_analysis import deterministic_investigation_fallback
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
            output = self.runner(
                question=investigation.user_question,
                df=df,
                data_context=run_context,
            )
            run.output = output if isinstance(output, dict) else {"raw_output": output}
            if self._has_runner_error(run.output):
                fallback = deterministic_investigation_fallback(investigation.user_question, df)
                if fallback:
                    fallback["fallback_from"] = run.output
                    run.output = fallback
                else:
                    raise RuntimeError(str(run.output.get("exec_error") or "Investigation runner failed."))
            update = agent_output_to_investigation_update(run.output)
            run.trace = update.trace
            run.status = InvestigationStatus.NEEDS_REVIEW
            run.finished_at = utc_now()

            self.store.add_run(investigation_id, run)
            artifact_ids: list[str] = []
            for artifact in update.artifacts:
                artifact.run_id = run.run_id
                self.store.add_artifact(investigation_id, artifact)
                if artifact.visibility == ArtifactVisibility.USER:
                    artifact_ids.append(artifact.artifact_id)
            for finding in update.findings:
                finding.run_id = run.run_id
                finding.evidence_artifact_ids = list(artifact_ids)
                self.store.add_finding(investigation_id, finding)
            if update.report:
                update.report.run_id = run.run_id
                update.report.question = update.report.question or investigation.user_question
                update.report.artifact_ids = list(artifact_ids)
                self.store.set_report(investigation_id, update.report)
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


def _question_with_data_context(question: str, data_context: dict[str, Any]) -> str:
    context_prompt = data_context.get("data_context_prompt")
    if not context_prompt:
        return question
    return f"{question}\n\nUse this data source context when planning the analysis:\n{context_prompt}"


def _unique_ids(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value and value not in seen:
            unique.append(value)
            seen.add(value)
    return unique


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
