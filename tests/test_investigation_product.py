from __future__ import annotations

import pytest
import pandas as pd

from source.product.adapter import agent_output_to_investigation_update
from source.product.investigation import Artifact, ArtifactType, InvestigationStatus
from source.product.service import InvestigationService
from source.product.store import InvestigationStore


def test_create_investigation() -> None:
    store = InvestigationStore()

    investigation = store.create_investigation("Why did conversion drop?")

    assert investigation.user_question == "Why did conversion drop?"
    assert investigation.title == "Why did conversion drop?"
    assert investigation.status == InvestigationStatus.DRAFT
    assert store.get_investigation(investigation.investigation_id) == investigation


def test_add_artifact() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Explain revenue growth")
    artifact = Artifact(
        artifact_type=ArtifactType.TEXT,
        title="Summary",
        content="Revenue grew in enterprise accounts.",
    )

    updated = store.add_artifact(investigation.investigation_id, artifact)

    assert updated.artifacts == [artifact]
    assert updated.updated_at >= updated.created_at


def test_adapter_handles_empty_output() -> None:
    update = agent_output_to_investigation_update({})

    assert update.artifacts == []
    assert update.findings == []
    assert update.report is None
    assert update.trace == []


def test_adapter_maps_summary_key_findings_generated_code() -> None:
    update = agent_output_to_investigation_update(
        {
            "structured_report": {
                "summary": "Conversion dropped in paid traffic.",
                "key_findings": ["Paid traffic CVR fell by 12%."],
                "limitations": ["No campaign spend data."],
            },
            "generated_code": "result = df.head()",
            "tool_timeline": [{"tool": "inspect_dataset_schema", "status": "ok"}],
        }
    )

    assert update.report is not None
    assert update.report.summary == "Conversion dropped in paid traffic."
    assert [finding.text for finding in update.findings] == [
        "Paid traffic CVR fell by 12%.",
        "No campaign spend data.",
    ]
    assert any(artifact.artifact_type == ArtifactType.PYTHON_CODE for artifact in update.artifacts)
    assert any(artifact.artifact_type == ArtifactType.REPORT for artifact in update.artifacts)
    assert any(artifact.artifact_type == ArtifactType.VALIDATION for artifact in update.artifacts)
    assert update.trace == [{"tool": "inspect_dataset_schema", "status": "ok"}]


def test_service_marks_failed_on_runner_exception() -> None:
    def failing_runner(**kwargs):
        raise RuntimeError("runner exploded")

    service = InvestigationService(runner=failing_runner)
    investigation = service.create_investigation("Why did churn spike?")

    updated = service.run_investigation(investigation.investigation_id, df=None)

    assert updated.status == InvestigationStatus.FAILED
    assert updated.runs[-1].status == InvestigationStatus.FAILED
    assert updated.runs[-1].error == "RuntimeError: runner exploded"
    assert updated.artifacts[-1].artifact_type == ArtifactType.VALIDATION
    assert "runner exploded" in updated.artifacts[-1].content


def test_service_marks_failed_on_runner_error_without_fallback() -> None:
    def error_runner(**kwargs):
        return {"exec_error": "Deep Agent did not produce an analytics tool result.", "critic_verdict": "ERROR"}

    service = InvestigationService(runner=error_runner)
    investigation = service.create_investigation("Why did churn spike?")

    updated = service.run_investigation(investigation.investigation_id, df=None)

    assert updated.status == InvestigationStatus.FAILED
    assert "analytics tool result" in updated.runs[-1].error


def test_service_uses_deterministic_fallback_for_profit_by_segment() -> None:
    def error_runner(**kwargs):
        return {"exec_error": "Deep Agent did not produce an analytics tool result.", "critic_verdict": "ERROR"}

    df = pd.DataFrame(
        {
            "Segment": ["Consumer", "Consumer", "Corporate"],
            "Profit": [10.0, 5.0, 30.0],
        }
    )
    service = InvestigationService(runner=error_runner)
    investigation = service.create_investigation("Почему прибыль отличается по сегментам клиентов?")

    updated = service.run_investigation(investigation.investigation_id, df=df)

    assert updated.status == InvestigationStatus.NEEDS_REVIEW
    assert updated.findings
    assert "Corporate" in updated.findings[0].text
    assert "причинно-следственную связь" in updated.report.limitations[0]
    assert any(artifact.artifact_type == ArtifactType.TABLE for artifact in updated.artifacts)
    assert updated.trace[0]["tool"] == "deterministic_pandas_fallback"


def test_service_returns_data_coverage_finding_when_requested_metric_missing() -> None:
    def error_runner(**kwargs):
        return {"exec_error": "Deep Agent did not produce an analytics tool result.", "critic_verdict": "ERROR"}

    df = pd.DataFrame(
        {
            "Segment": ["Consumer", "Corporate"],
            "Sales": [100.0, 250.0],
        }
    )
    service = InvestigationService(runner=error_runner)
    investigation = service.create_investigation("Почему прибыль отличается по сегментам клиентов?")

    updated = service.run_investigation(investigation.investigation_id, df=df)

    assert updated.status == InvestigationStatus.NEEDS_REVIEW
    assert updated.findings
    assert "no matching `profit` column" in updated.findings[0].text
    assert updated.report is not None
    assert "Sales" in updated.report.next_steps[-1]
    assert updated.trace[0]["tool"] == "data_coverage_check"


def test_service_fallback_handles_sales_by_ship_mode() -> None:
    def error_runner(**kwargs):
        return {"exec_error": "Deep Agent did not produce an analytics tool result.", "critic_verdict": "ERROR"}

    df = pd.DataFrame(
        {
            "Ship Mode": ["Second Class", "Standard Class", "Second Class"],
            "Sales": [100.0, 250.0, 50.0],
        }
    )
    service = InvestigationService(runner=error_runner)
    investigation = service.create_investigation("Как режим доставки связан с объемом продаж?")

    updated = service.run_investigation(investigation.investigation_id, df=df)

    assert updated.status == InvestigationStatus.NEEDS_REVIEW
    assert updated.findings
    assert "`Ship Mode`" in updated.findings[0].text
    assert "`Sales`" in updated.findings[0].text
    table_artifact = next(artifact for artifact in updated.artifacts if artifact.artifact_type == ArtifactType.TABLE)
    assert isinstance(table_artifact.content, list)
    assert table_artifact.content[0]["Ship Mode"] == "Standard Class"
    assert updated.trace[0]["tool"] == "deterministic_pandas_fallback"
