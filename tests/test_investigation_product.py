from __future__ import annotations

import pytest
import pandas as pd

from source.product.adapter import agent_output_to_investigation_update
from source.product.investigation import Artifact, ArtifactType, InvestigationStatus
from source.product.service import InvestigationService
from source.product.store import InvestigationStore

FORBIDDEN_ANALYSIS_NARRATION = (
    "to identify",
    "you can compare",
    "a useful next step",
    "should focus",
    "to investigate",
    "can be analyzed",
    "supports analysis",
    "potential grouping",
    "next analytical move",
    "useful analysis",
    "should be compared",
)

def assert_no_methodology_narration(text: str) -> None:
    lowered = text.lower()
    for phrase in FORBIDDEN_ANALYSIS_NARRATION:
        assert phrase not in lowered

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

def test_service_fallback_handles_metric_by_dimension() -> None:
    def error_runner(**kwargs):
        return {"exec_error": "Deep Agent did not produce an analytics tool result.", "critic_verdict": "ERROR"}

    df = pd.DataFrame(
        {
            "dimension_label": ["Group A", "Group B", "Group A"],
            "metric_value": [100.0, 250.0, 50.0],
        }
    )
    service = InvestigationService(runner=error_runner)
    investigation = service.create_investigation("Как группы связаны со значением метрики?")

    updated = service.run_investigation(investigation.investigation_id, df=df)

    assert updated.status == InvestigationStatus.NEEDS_REVIEW
    assert updated.findings
    assert "`dimension_label`" in updated.findings[0].text
    assert "`metric_value`" in updated.findings[0].text
    table_artifact = next(artifact for artifact in updated.artifacts if artifact.artifact_type == ArtifactType.TABLE)
    assert isinstance(table_artifact.content, list)
    assert table_artifact.content[0]["dimension_label"] == "Group B"
    assert updated.trace[0]["tool"] == "deterministic_pandas_fallback"

