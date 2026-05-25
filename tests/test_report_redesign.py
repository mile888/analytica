from __future__ import annotations

from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

from source.product.exporter import export_shareable_report_pdf
from source.product.investigation import (
    Artifact,
    ArtifactType,
    Finding,
    FindingStatus,
    Investigation,
    InvestigationMessage,
    InvestigationMessageRole,
    InvestigationMessageType,
)
from source.product.report_builder import build_shareable_report
from source.product.report_artifacts import artifact_report_snapshot


def _table(title: str, artifact_id: str | None = None) -> Artifact:
    artifact = Artifact(
        artifact_type=ArtifactType.TABLE,
        title=title,
        content=[{"City": "LA", "Sales": 10}, {"City": "SF", "Sales": 7}],
        metadata={
            "selected_for_report": True,
            "metric": "Sales",
            "dimension": "City",
            "dataset_id": "ds_sales",
            "dataset_ids": ["ds_sales"],
            "branch_id": "table::ds_sales::Sales::City",
            "created_from_query": "top cities by sales",
        },
    )
    if artifact_id:
        artifact.artifact_id = artifact_id
    return artifact


def _chart(title: str, artifact_id: str | None = None) -> Artifact:
    artifact = Artifact(
        artifact_type=ArtifactType.CHART,
        title=title,
        content={
            "chart_type": "bar",
            "x": "City",
            "y": "total",
            "rows": [{"City": "LA", "total": 10}, {"City": "SF", "total": 7}],
        },
        metadata={
            "selected_for_report": True,
            "metric": "Sales",
            "dimension": "City",
            "dataset_id": "ds_sales",
            "dataset_ids": ["ds_sales"],
            "branch_id": "grouped::ds_sales::Sales::City::sum::",
            "created_from_query": "выручка по городам",
        },
    )
    if artifact_id:
        artifact.artifact_id = artifact_id
    return artifact


def _investigation() -> Investigation:
    investigation = Investigation(title="Sales investigation", user_question="выручка по городам")
    investigation.linked_data_source_ids = ["ds_sales", "ds_salary"]
    investigation.findings = [
        Finding(
            title="Revenue concentration",
            text="LA leads revenue.",
            status=FindingStatus.PROPOSED,
            confidence=0.9,
            metadata={
                "conclusion": "LA leads `Sales` by city.",
                "confidence": "High",
                "evidence_reason": "Supported by selected city revenue chart.",
                "limitation": "Small sample sizes may distort city ranking.",
                "recommended_validation": "Check volume and outliers by city.",
            },
        )
    ]
    investigation.artifacts = [_chart("Histogram of Sales in LA", "chart_hist"), _chart("Adjusted city ranking", "chart_adjusted"), _table("Top city sales table", "table_top_cities")]
    return investigation


def _messages(investigation: Investigation) -> list[InvestigationMessage]:
    return [
        InvestigationMessage(investigation_id=investigation.investigation_id, role=InvestigationMessageRole.USER, message_type=InvestigationMessageType.QUESTION, content="выручка по городам", message_id="q1", run_id="r1"),
        InvestigationMessage(investigation_id=investigation.investigation_id, role=InvestigationMessageRole.ASSISTANT, message_type=InvestigationMessageType.RUN_SUMMARY, content="Revenue by city was analyzed.", run_id="r1"),
        InvestigationMessage(investigation_id=investigation.investigation_id, role=InvestigationMessageRole.USER, message_type=InvestigationMessageType.FOLLOW_UP, content="Explain this chart", message_id="q2", run_id="r2", metadata={"artifact_id": "chart_hist"}),
        InvestigationMessage(investigation_id=investigation.investigation_id, role=InvestigationMessageRole.ASSISTANT, message_type=InvestigationMessageType.RUN_SUMMARY, content="The chart shows LA remains ahead after filtering.", run_id="r2"),
        InvestigationMessage(investigation_id=investigation.investigation_id, role=InvestigationMessageRole.USER, message_type=InvestigationMessageType.FOLLOW_UP, content="Show the table too", message_id="q3", run_id="r3"),
        InvestigationMessage(investigation_id=investigation.investigation_id, role=InvestigationMessageRole.ASSISTANT, message_type=InvestigationMessageType.RUN_SUMMARY, content="The table was added as report-ready evidence.", run_id="r3"),
    ]


def test_structured_report_ingests_history_findings_limitations_and_chronology() -> None:
    investigation = _investigation()
    report = build_shareable_report(investigation, messages=_messages(investigation))

    titles = [section.title for section in report.sections]
    for title in ["Investigation summary", "Key findings", "Analytical workflow", "Complete Q&A transcript", "Visual analysis", "Limitations", "Conclusions"]:
        assert title in titles
    assert "Analytical workflow" in titles
    workflow = next(section for section in report.sections if section.title == "Analytical workflow")
    transcript = next(section for section in report.sections if section.title == "Complete Q&A transcript")
    limitations = next(section for section in report.sections if section.title == "Limitations")

    assert "выручка по городам" in workflow.content
    assert "Explain this chart" in workflow.content
    assert "3. Question: Show the table too" in transcript.content
    assert "Answer: The table was added as report-ready evidence." in transcript.content
    assert "Small sample sizes" in limitations.content
    assert report.dataset_ids == ["ds_sales", "ds_salary"]
    assert "ds_sales" in report.metadata["dataset_ids"]
    assert report.included_question_ids == ["q1", "q2", "q3"]


def test_selected_artifacts_snapshot_images_and_reload_metadata_are_stable() -> None:
    investigation = _investigation()
    report = build_shareable_report(investigation, messages=_messages(investigation))
    visual = next(section for section in report.sections if section.section_type == "visual_analysis")
    snapshots = visual.metadata["artifact_snapshots"]

    assert report.source_artifact_ids == ["chart_hist", "chart_adjusted", "table_top_cities"]
    assert [item["title"] for item in snapshots] == ["Histogram of Sales in LA", "Adjusted city ranking", "Top city sales table"]
    assert all(Path(item["image_path"]).exists() for item in snapshots if item["artifact_type"] == "chart")
    assert any(item["artifact_type"] == "table" for item in snapshots)
    assert all(item["dataset_id"] == "ds_sales" for item in snapshots)
    assert "Originating question" in snapshots[0]["description"]


def test_pdf_export_includes_multiple_chart_image_pages_and_is_deterministic_enough() -> None:
    investigation = _investigation()
    report = build_shareable_report(investigation, messages=_messages(investigation))

    first = export_shareable_report_pdf(report, investigation.artifacts)
    second = export_shareable_report_pdf(report, investigation.artifacts)
    reader = PdfReader(BytesIO(first))

    assert first[:4] == b"%PDF"
    assert len(reader.pages) >= 3
    assert abs(len(first) - len(second)) < 500


def test_artifact_snapshot_preserves_required_report_fields() -> None:
    artifact = _chart("Sales by city")
    snapshot = artifact_report_snapshot(artifact, image_path="/tmp/chart.png")

    for key in [
        "artifact_id",
        "artifact_type",
        "chart_type",
        "dataset_id",
        "title",
        "description",
        "image_path",
        "image_bytes_reference",
        "filters",
        "metric",
        "dimension",
        "created_from_query",
        "branch_id",
    ]:
        assert key in snapshot
