from __future__ import annotations

from source.product.exporter import (
    build_artifact_summary,
    export_investigation_html,
    export_investigation_markdown,
)
from source.product.investigation import (
    Artifact,
    ArtifactType,
    ArtifactVisibility,
    DecisionReport,
    Finding,
    FindingStatus,
    Investigation,
)


def test_markdown_export_works_with_empty_investigation() -> None:
    investigation = Investigation(title="Empty", user_question="Question?")

    markdown = export_investigation_markdown(investigation)

    assert "# Empty" in markdown
    assert "_No DecisionReport yet._" in markdown
    assert "_No findings yet._" in markdown


def test_markdown_export_includes_report_fields() -> None:
    investigation = Investigation(title="Report", user_question="Question?")
    investigation.report = DecisionReport(
        question="Question?",
        answer="Answer",
        key_findings=["Finding"],
        evidence=["Evidence"],
        limitations=["Limitation"],
        next_steps=["Next step"],
    )

    markdown = export_investigation_markdown(investigation)

    assert "### Answer" in markdown
    assert "Answer" in markdown
    assert "- Finding" in markdown
    assert "- Evidence" in markdown
    assert "- Limitation" in markdown
    assert "- Next step" in markdown


def test_markdown_export_excludes_technical_artifacts_by_default() -> None:
    investigation = Investigation(title="Artifacts", user_question="Question?")
    investigation.artifacts = [
        Artifact(artifact_type=ArtifactType.TEXT, title="User note", content="Visible"),
        Artifact(
            artifact_type=ArtifactType.PYTHON_CODE,
            title="Generated Python",
            content="result = df.head()",
            visibility=ArtifactVisibility.TECHNICAL,
        ),
    ]

    markdown = export_investigation_markdown(investigation)

    assert "User note" in markdown
    assert "Generated Python" not in markdown
    assert "result = df.head()" not in markdown


def test_markdown_export_includes_technical_artifacts_when_requested() -> None:
    investigation = Investigation(title="Artifacts", user_question="Question?")
    investigation.artifacts = [
        Artifact(
            artifact_type=ArtifactType.SQL,
            title="SQL context",
            content="SELECT * FROM data",
            visibility=ArtifactVisibility.TECHNICAL,
        )
    ]

    markdown = export_investigation_markdown(investigation, include_technical=True)

    assert "## Technical appendix" in markdown
    assert "SQL context" in markdown
    assert "SELECT * FROM data" in markdown


def test_hidden_artifacts_are_never_exported() -> None:
    investigation = Investigation(title="Artifacts", user_question="Question?")
    investigation.artifacts = [
        Artifact(
            artifact_type=ArtifactType.TEXT,
            title="Hidden raw output",
            content="secret",
            visibility=ArtifactVisibility.HIDDEN,
            pinned=True,
        )
    ]

    markdown = export_investigation_markdown(investigation, include_technical=True)
    summary = build_artifact_summary(investigation, include_technical=True)

    assert "secret" not in markdown
    assert summary == []


def test_pinned_artifacts_appear_before_unpinned_artifacts() -> None:
    investigation = Investigation(title="Artifacts", user_question="Question?")
    investigation.artifacts = [
        Artifact(artifact_type=ArtifactType.TEXT, title="Unpinned", content="B", pinned=False),
        Artifact(artifact_type=ArtifactType.TEXT, title="Pinned", content="A", pinned=True),
    ]

    markdown = export_investigation_markdown(investigation)

    assert markdown.index("### Pinned") < markdown.index("### Unpinned")


def test_html_export_escapes_unsafe_content() -> None:
    investigation = Investigation(title="<script>alert(1)</script>", user_question="<b>Question?</b>")
    investigation.report = DecisionReport(answer="<img src=x onerror=alert(1)>")

    html = export_investigation_html(investigation)

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<img src=x onerror=alert(1)>" not in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html


def test_markdown_export_orders_accepted_findings_before_proposed() -> None:
    investigation = Investigation(title="Findings", user_question="Question?")
    investigation.findings = [
        Finding(title="Proposed", text="Proposed finding", status=FindingStatus.PROPOSED),
        Finding(title="Accepted", text="Accepted finding", status=FindingStatus.ACCEPTED),
    ]

    markdown = export_investigation_markdown(investigation)

    assert markdown.index("### Accepted") < markdown.index("### Proposed")
