from __future__ import annotations

from fastapi.testclient import TestClient

from source.api.app import app
from source.api.deps import set_store_for_testing
from source.product.cross_investigation import extract_cross_investigation_patterns
from source.product.investigation import Artifact, ArtifactType, Finding, Investigation
from source.product.organizational_workflows import (
    build_organizational_playbooks,
    evidence_requirements_for_finding,
    evaluate_workflow_progress,
    recommend_organizational_workflow,
    report_standard_for_template,
    review_investigation,
    validation_expectations_for_investigation,
)
from source.product.report_builder import build_shareable_report
from source.product.store import InvestigationStore


def _grouped_investigation() -> Investigation:
    investigation = Investigation(title="Generic performance review", user_question="Compare metric by segment")
    investigation.linked_data_source_ids = ["ds_generic"]
    finding = Finding(
        title="Metric varies by segment",
        text="The metric differs materially across segments.",
        evidence_artifact_ids=["artifact_chart"],
        metadata={
            "analysis_type": "grouped_metric",
            "conclusion": "The metric differs materially across segments.",
            "evidence_strength": "high",
            "supporting_artifact_ids": ["artifact_chart"],
            "supporting_dimensions": ["segment_label"],
            "related_metrics": ["metric_value"],
        },
    )
    investigation.findings = [finding]
    investigation.artifacts = [
        Artifact(
            artifact_id="artifact_chart",
            artifact_type=ArtifactType.CHART,
            title="Metric by Segment",
            content={"chart_type": "bar", "metric": "metric_value", "x": "segment_label"},
        )
    ]
    return investigation


def test_workflow_recommendation_and_progress_are_dataset_agnostic() -> None:
    investigation = _grouped_investigation()
    patterns = extract_cross_investigation_patterns([investigation])

    workflow = recommend_organizational_workflow(investigation, patterns)
    progress = evaluate_workflow_progress(investigation, workflow)

    assert workflow.workflow_id == "exploratory_investigation"
    assert workflow.stages
    assert any(item.status == "complete" for item in progress)
    assert "Salary_LPA" not in workflow.description


def test_validation_expectations_and_evidence_requirements_detect_missing_support() -> None:
    finding = Finding(
        text="A metric appears unstable across groups.",
        metadata={"analysis_type": "grouped_metric", "evidence_strength": "low"},
    )
    investigation = Investigation(title="Weak evidence", user_question="Compare metric")
    investigation.findings = [finding]

    expectations = validation_expectations_for_investigation(investigation)
    requirements = evidence_requirements_for_finding(finding)

    assert any(item.expectation_id == "chart_or_table_evidence" for item in expectations)
    assert any(item.status == "missing" for item in requirements)
    assert any("chart" in item.message.lower() or "table" in item.message.lower() for item in requirements)


def test_review_flow_is_guidance_not_hard_blocking() -> None:
    investigation = Investigation(title="Early", user_question="What should we inspect?")

    review = review_investigation(investigation)

    assert review.status in {"developing", "needs_validation"}
    assert review.evidence_quality == "early"
    assert review.suggested_next_step
    assert review.checkpoints


def test_organizational_playbooks_follow_semantic_patterns() -> None:
    investigation = _grouped_investigation()
    patterns = extract_cross_investigation_patterns([investigation])

    playbooks = build_organizational_playbooks(patterns)

    assert playbooks
    assert any("segmentation" in playbook.title.lower() or "metric" in playbook.title.lower() for playbook in playbooks)
    assert all("careers" not in " ".join(playbook.analytical_sequence).lower() for playbook in playbooks)


def test_report_standard_reuse_improves_report_guidance() -> None:
    investigation = _grouped_investigation()
    patterns = extract_cross_investigation_patterns([investigation])
    review = review_investigation(investigation, patterns)
    standard = report_standard_for_template("executive_summary")

    report = build_shareable_report(
        investigation,
        analytical_patterns=patterns,
        organizational_review=review,
        report_standard=standard,
    )
    content = "\n".join(section.content for section in report.sections)

    assert any(section.title == "Evidence standards" for section in report.sections)
    assert "evidence" in content.lower()
    assert "workflow" not in content.lower()


def test_workflow_guidance_api_returns_review_payload() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Compare metric by segment")
    store.add_finding(
        investigation.investigation_id,
        Finding(text="Metric differs by segment.", metadata={"analysis_type": "grouped_metric"}),
    )
    set_store_for_testing(store)
    client = TestClient(app)

    response = client.get(f"/investigations/{investigation.investigation_id}/workflow-guidance")

    assert response.status_code == 200
    payload = response.json()
    assert payload["workflow"]["title"]
    assert payload["review"]["summary"]
    assert payload["validation_expectations"]
