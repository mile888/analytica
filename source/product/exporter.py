from __future__ import annotations

import html
import json
from typing import Any

from source.product.investigation import (
    Artifact,
    ArtifactType,
    ArtifactVisibility,
    Finding,
    FindingStatus,
    Investigation,
    ReportComment,
    ShareableReport,
)


def build_artifact_summary(investigation: Investigation, include_technical: bool = False) -> list[dict[str, Any]]:
    artifacts = [
        artifact
        for artifact in investigation.artifacts
        if artifact.visibility != ArtifactVisibility.HIDDEN
        and (artifact.visibility == ArtifactVisibility.USER or include_technical)
    ]
    artifacts.sort(key=lambda item: (not item.pinned, item.created_at.isoformat(), item.title))
    return [
        {
            "id": artifact.artifact_id,
            "title": artifact.title,
            "type": artifact.artifact_type.value,
            "visibility": artifact.visibility.value,
            "pinned": artifact.pinned,
            "content": artifact.content,
            "created_at": artifact.created_at.isoformat(),
        }
        for artifact in artifacts
    ]


def export_investigation_markdown(investigation: Investigation, include_technical: bool = False) -> str:
    lines: list[str] = [
        f"# {investigation.title}",
        "",
        f"**Status:** `{investigation.status.value}`",
        f"**Question:** {investigation.user_question}",
        f"**Created:** {investigation.created_at.isoformat()}",
        f"**Updated:** {investigation.updated_at.isoformat()}",
        "",
    ]

    report = investigation.report
    if report:
        lines.extend(["## DecisionReport", ""])
        _append_md_section(lines, "Answer", [report.answer or report.summary])
        _append_md_section(lines, "Key findings", report.key_findings, bullet=True)
        _append_md_section(lines, "Evidence", report.evidence, bullet=True)
        _append_md_section(lines, "Limitations", report.limitations, bullet=True)
        _append_md_section(lines, "Next steps", report.next_steps, bullet=True)
    else:
        lines.extend(["## DecisionReport", "", "_No DecisionReport yet._", ""])

    accepted = [item for item in investigation.findings if item.status == FindingStatus.ACCEPTED]
    proposed = [item for item in investigation.findings if item.status == FindingStatus.PROPOSED]
    rejected = [item for item in investigation.findings if item.status == FindingStatus.REJECTED]
    lines.extend(["## Findings", ""])
    if accepted or proposed or rejected:
        for finding in accepted + proposed + rejected:
            lines.extend(_finding_markdown(finding))
    else:
        lines.extend(["_No findings yet._", ""])

    user_artifacts = [
        artifact
        for artifact in investigation.artifacts
        if artifact.visibility == ArtifactVisibility.USER
    ]
    user_artifacts.sort(key=lambda item: (not item.pinned, item.created_at.isoformat(), item.title))
    lines.extend(["## Artifacts", ""])
    if user_artifacts:
        for artifact in user_artifacts:
            lines.extend(_artifact_markdown(artifact))
    else:
        lines.extend(["_No user-facing artifacts._", ""])

    if include_technical:
        technical_artifacts = [
            artifact
            for artifact in investigation.artifacts
            if artifact.visibility == ArtifactVisibility.TECHNICAL
        ]
        technical_artifacts.sort(key=lambda item: (not item.pinned, item.created_at.isoformat(), item.title))
        lines.extend(["## Technical appendix", ""])
        if technical_artifacts:
            for artifact in technical_artifacts:
                lines.extend(_artifact_markdown(artifact))
        else:
            lines.extend(["_No technical artifacts._", ""])

    return "\n".join(lines).strip() + "\n"


def export_investigation_html(investigation: Investigation, include_technical: bool = False) -> str:
    title = html.escape(investigation.title)
    body: list[str] = [
        "<!doctype html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8">',
        f"<title>{title}</title>",
        "<style>",
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:920px;margin:40px auto;padding:0 24px;line-height:1.55;color:#17202a}",
        "h1,h2,h3{line-height:1.2} code{background:#f4f6f8;padding:2px 5px;border-radius:4px} pre{background:#f4f6f8;padding:14px;border-radius:8px;overflow:auto}",
        ".meta{color:#5f6b7a}.finding,.artifact{border:1px solid #d9dee7;border-radius:8px;padding:14px;margin:12px 0}.badge{font-size:12px;color:#445;background:#eef2f6;padding:2px 6px;border-radius:999px}",
        "table{border-collapse:collapse;width:100%;margin:8px 0}th,td{border:1px solid #d9dee7;padding:6px 8px;text-align:left}th{background:#f4f6f8}",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{title}</h1>",
        f"<p class='meta'><strong>Status:</strong> {html.escape(investigation.status.value)}<br>",
        f"<strong>Question:</strong> {html.escape(investigation.user_question)}<br>",
        f"<strong>Created:</strong> {html.escape(investigation.created_at.isoformat())}<br>",
        f"<strong>Updated:</strong> {html.escape(investigation.updated_at.isoformat())}</p>",
    ]

    report = investigation.report
    body.append("<h2>DecisionReport</h2>")
    if report:
        _append_html_section(body, "Answer", [report.answer or report.summary])
        _append_html_section(body, "Key findings", report.key_findings, bullet=True)
        _append_html_section(body, "Evidence", report.evidence, bullet=True)
        _append_html_section(body, "Limitations", report.limitations, bullet=True)
        _append_html_section(body, "Next steps", report.next_steps, bullet=True)
    else:
        body.append("<p><em>No DecisionReport yet.</em></p>")

    findings = sorted(
        investigation.findings,
        key=lambda item: (_finding_sort_rank(item.status), item.created_at.isoformat()),
    )
    body.append("<h2>Findings</h2>")
    if findings:
        for finding in findings:
            body.append("<div class='finding'>")
            body.append(
                f"<h3>{html.escape(finding.title or 'Finding')} <span class='badge'>{html.escape(finding.status.value)}</span></h3>"
            )
            body.append(f"<p>{html.escape(finding.text)}</p>")
            body.append("</div>")
    else:
        body.append("<p><em>No findings yet.</em></p>")

    body.append("<h2>Artifacts</h2>")
    artifact_summaries = build_artifact_summary(investigation, include_technical=include_technical)
    visible_artifacts = [
        item
        for item in artifact_summaries
        if item["visibility"] == ArtifactVisibility.USER.value
        or (include_technical and item["visibility"] == ArtifactVisibility.TECHNICAL.value)
    ]
    if visible_artifacts:
        for artifact in visible_artifacts:
            body.append("<div class='artifact'>")
            body.append(
                f"<h3>{html.escape(artifact['title'] or 'Artifact')} <span class='badge'>{html.escape(artifact['type'])}</span></h3>"
            )
            body.append(_artifact_content_html(artifact["content"]))
            body.append("</div>")
    else:
        body.append("<p><em>No exportable artifacts.</em></p>")

    body.extend(["</body>", "</html>"])
    return "\n".join(body)


def export_shareable_report_markdown(
    report: ShareableReport,
    include_comments: bool = False,
    comments: list[ReportComment] | None = None,
) -> str:
    lines: list[str] = [
        f"# {report.title}",
        "",
        f"**Status:** `{report.status.value}`",
        f"**Approval:** `{report.approval_status.value}`",
        f"**Template:** `{report.template.value}`",
        f"**Version:** `{report.version}`",
        f"**Created:** {report.created_at.isoformat()}",
        f"**Updated:** {report.updated_at.isoformat()}",
    ]
    if report.approved_by:
        lines.append(f"**Approved by:** {report.approved_by}")
    if report.approved_at:
        lines.append(f"**Approved at:** {report.approved_at.isoformat()}")
    lines.append("")
    sections = sorted(report.sections, key=lambda item: (item.order, item.title))
    if sections:
        for section in sections:
            lines.extend([f"## {section.title}", "", section.content or "_No content._", ""])
    else:
        lines.extend(["_No sections yet._", ""])
    if include_comments:
        lines.extend(["## Review comments", ""])
        visible_comments = comments or []
        if visible_comments:
            section_titles = {section.section_id: section.title for section in report.sections}
            for comment in visible_comments:
                lines.extend(
                    [
                        f"### {section_titles.get(comment.section_id, 'Section comment')}",
                        "",
                        f"**Status:** `{comment.status.value}`",
                        f"**Author:** {comment.author}",
                        f"**Created:** {comment.created_at.isoformat()}",
                        "",
                        comment.text,
                        "",
                    ]
                )
        else:
            lines.extend(["_No review comments._", ""])
    return "\n".join(lines).strip() + "\n"


def export_shareable_report_html(
    report: ShareableReport,
    include_comments: bool = False,
    comments: list[ReportComment] | None = None,
) -> str:
    title = html.escape(report.title)
    body: list[str] = [
        "<!doctype html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8">',
        f"<title>{title}</title>",
        "<style>",
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:880px;margin:40px auto;padding:0 24px;line-height:1.58;color:#17202a}",
        "h1,h2{line-height:1.2}.meta{color:#5f6b7a}section{border-bottom:1px solid #e5e9f0;padding:14px 0}pre{white-space:pre-wrap;background:#f4f6f8;border-radius:8px;padding:12px}",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{title}</h1>",
        f"<p class='meta'><strong>Status:</strong> {html.escape(report.status.value)}<br>",
        f"<strong>Approval:</strong> {html.escape(report.approval_status.value)}<br>",
        f"<strong>Template:</strong> {html.escape(report.template.value)}<br>",
        f"<strong>Version:</strong> {report.version}<br>",
        f"<strong>Created:</strong> {html.escape(report.created_at.isoformat())}<br>",
        f"<strong>Updated:</strong> {html.escape(report.updated_at.isoformat())}",
    ]
    if report.approved_by:
        body.append(f"<br><strong>Approved by:</strong> {html.escape(report.approved_by)}")
    if report.approved_at:
        body.append(f"<br><strong>Approved at:</strong> {html.escape(report.approved_at.isoformat())}")
    body.append("</p>")
    sections = sorted(report.sections, key=lambda item: (item.order, item.title))
    if sections:
        for section in sections:
            body.append("<section>")
            body.append(f"<h2>{html.escape(section.title)}</h2>")
            body.append(_section_content_html(section.content))
            body.append("</section>")
    else:
        body.append("<p><em>No sections yet.</em></p>")
    if include_comments:
        body.append("<h2>Review comments</h2>")
        visible_comments = comments or []
        if visible_comments:
            section_titles = {section.section_id: section.title for section in report.sections}
            for comment in visible_comments:
                body.append("<section>")
                body.append(f"<h3>{html.escape(section_titles.get(comment.section_id, 'Section comment'))}</h3>")
                body.append(
                    f"<p class='meta'><strong>Status:</strong> {html.escape(comment.status.value)}<br>"
                    f"<strong>Author:</strong> {html.escape(comment.author)}<br>"
                    f"<strong>Created:</strong> {html.escape(comment.created_at.isoformat())}</p>"
                )
                body.append(f"<p>{html.escape(comment.text)}</p>")
                body.append("</section>")
        else:
            body.append("<p><em>No review comments.</em></p>")
    body.extend(["</body>", "</html>"])
    return "\n".join(body)


def _append_md_section(lines: list[str], title: str, values: list[str], bullet: bool = False) -> None:
    if not values:
        return
    lines.extend([f"### {title}", ""])
    if bullet:
        lines.extend(f"- {value}" for value in values)
    else:
        lines.extend(str(value) for value in values if value)
    lines.append("")


def _append_html_section(body: list[str], title: str, values: list[str], bullet: bool = False) -> None:
    if not values:
        return
    body.append(f"<h3>{html.escape(title)}</h3>")
    if bullet:
        body.append("<ul>")
        body.extend(f"<li>{html.escape(str(value))}</li>" for value in values)
        body.append("</ul>")
    else:
        body.extend(f"<p>{html.escape(str(value))}</p>" for value in values if value)


def _finding_markdown(finding: Finding) -> list[str]:
    lines = [f"### {finding.title or 'Finding'}", "", f"**Status:** `{finding.status.value}`", "", finding.text, ""]
    if finding.evidence:
        lines.extend(["Evidence:", ""])
        lines.extend(f"- {item}" for item in finding.evidence)
        lines.append("")
    return lines


def _artifact_markdown(artifact: Artifact) -> list[str]:
    lines = [
        f"### {artifact.title or 'Artifact'}",
        "",
        f"**Type:** `{artifact.artifact_type.value}`  ",
        f"**Visibility:** `{artifact.visibility.value}`  ",
        f"**Pinned:** `{artifact.pinned}`",
        "",
    ]
    lines.extend(_artifact_content_markdown(artifact))
    lines.append("")
    return lines


def _artifact_content_markdown(artifact: Artifact) -> list[str]:
    content = artifact.content
    if artifact.artifact_type == ArtifactType.TABLE and isinstance(content, list) and all(isinstance(row, dict) for row in content):
        return _markdown_table(content)
    if isinstance(content, (dict, list)):
        return ["```json", json.dumps(content, ensure_ascii=False, indent=2, default=str), "```"]
    if artifact.artifact_type == ArtifactType.PYTHON_CODE:
        return ["```python", str(content or ""), "```"]
    if artifact.artifact_type == ArtifactType.SQL:
        return ["```sql", str(content or ""), "```"]
    return [str(content or "")]


def _markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["_Empty table._"]
    columns = list(rows[0].keys())
    lines = [
        "| " + " | ".join(str(column) for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_md_cell(row.get(column)) for column in columns) + " |")
    return lines


def _md_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _artifact_content_html(content: Any) -> str:
    if isinstance(content, list) and all(isinstance(row, dict) for row in content):
        if not content:
            return "<p><em>Empty table.</em></p>"
        columns = list(content[0].keys())
        rows = ["<table><thead><tr>"]
        rows.extend(f"<th>{html.escape(str(column))}</th>" for column in columns)
        rows.append("</tr></thead><tbody>")
        for row in content:
            rows.append("<tr>")
            rows.extend(f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in columns)
            rows.append("</tr>")
        rows.append("</tbody></table>")
        return "".join(rows)
    if isinstance(content, (dict, list)):
        return f"<pre>{html.escape(json.dumps(content, ensure_ascii=False, indent=2, default=str))}</pre>"
    return f"<p>{html.escape(str(content or ''))}</p>"


def _section_content_html(content: str) -> str:
    escaped = html.escape(content or "")
    if not escaped:
        return "<p><em>No content.</em></p>"
    paragraphs = escaped.split("\n\n")
    rendered: list[str] = []
    for paragraph in paragraphs:
        if paragraph.startswith("- "):
            items = [line[2:] for line in paragraph.splitlines() if line.startswith("- ")]
            rendered.append("<ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>")
        else:
            rendered.append(f"<p>{paragraph.replace(chr(10), '<br>')}</p>")
    return "\n".join(rendered)


def _finding_sort_rank(status: FindingStatus) -> int:
    return {
        FindingStatus.ACCEPTED: 0,
        FindingStatus.PROPOSED: 1,
        FindingStatus.REJECTED: 2,
    }.get(status, 3)
