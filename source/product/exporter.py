from __future__ import annotations

import html
import json
from io import BytesIO
import textwrap
from typing import Any

from source.product.investigation import (
    Artifact,
    ArtifactType,
    ArtifactVisibility,
    FinalReportSnapshot,
    Finding,
    FindingStatus,
    Investigation,
    ReportComment,
    ReportSection,
    ShareableReport,
    ShareableReportStatus,
    ShareableReportTemplate,
    ReportApprovalStatus,
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
            metadata = finding.metadata or {}
            conclusion = _metadata_text(metadata, "conclusion") or finding.text
            body.append("<div class='finding'>")
            body.append(
                f"<h3>{html.escape(finding.title or 'Finding')} <span class='badge'>{html.escape(finding.status.value)}</span></h3>"
            )
            body.append(f"<p>{html.escape(conclusion)}</p>")
            structured = [
                ("Confidence", _metadata_text(metadata, "confidence_reason")),
                ("Evidence", _metadata_text(metadata, "evidence_reason")),
                ("Implication", _metadata_text(metadata, "business_implication")),
                ("Limitation", _metadata_text(metadata, "limitation")),
                ("Recommended validation", _metadata_text(metadata, "recommended_validation")),
            ]
            for label, value in structured:
                if value:
                    body.append(f"<p><strong>{html.escape(label)}:</strong> {html.escape(value)}</p>")
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
        "| Field | Value |",
        "| --- | --- |",
        f"| Status | `{report.status.value}` |",
        f"| Approval | `{report.approval_status.value}` |",
        f"| Template | `{report.template.value}` |",
        f"| Version | {report.version} |",
        f"| Created | {report.created_at.isoformat()} |",
        f"| Updated | {report.updated_at.isoformat()} |",
    ]
    if report.approved_by:
        lines.append(f"| Approved by | {report.approved_by} |")
    if report.approved_at:
        lines.append(f"| Approved at | {report.approved_at.isoformat()} |")
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


def export_shareable_report_txt(report: ShareableReport) -> str:
    lines = _txt_header(report.title)
    lines.extend(
        [
            f"Status: {_title_value(report.status.value)}",
            f"Approval: {_title_value(report.approval_status.value)}",
            f"Template: {_title_value(report.template.value)}",
            f"Version: {report.version}",
            f"Created: {report.created_at.isoformat()}",
            f"Updated: {report.updated_at.isoformat()}",
        ]
    )
    if report.approved_by:
        lines.append(f"Approved by: {report.approved_by}")
    if report.approved_at:
        lines.append(f"Approved at: {report.approved_at.isoformat()}")
    lines.append("")
    sections = sorted(report.sections, key=lambda item: (item.order, item.title))
    if sections:
        for section in sections:
            lines.extend(_txt_section(section.title, section.content or "No content."))
    else:
        lines.extend(_txt_section("Report", "No sections yet."))
    return "\n".join(lines).strip() + "\n"


def export_shareable_report_pdf(report: ShareableReport, artifacts: list[Artifact] | None = None) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    output = BytesIO()
    with PdfPages(output) as pdf:
        _append_report_text_pages(pdf, plt, report)
        for artifact in _pdf_artifacts(report, artifacts or []):
            if artifact.artifact_type == ArtifactType.CHART:
                fig = _chart_artifact_figure(plt, artifact)
            elif artifact.artifact_type == ArtifactType.TABLE:
                fig = _table_artifact_figure(plt, artifact)
            else:
                fig = None
            if fig is None:
                continue
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return output.getvalue()


def export_final_report_snapshot_txt(snapshot: FinalReportSnapshot) -> str:
    if snapshot.txt_content:
        return snapshot.txt_content
    report = _report_from_snapshot_json(snapshot.source_report_json, snapshot)
    if report:
        return export_shareable_report_txt(report)
    lines = _txt_header(snapshot.title)
    lines.extend(
        [
            f"Status: {_title_value(snapshot.status.value)}",
            f"Approval: {_title_value(snapshot.approval_status.value)}",
            f"Report version: {snapshot.report_version}",
            f"Created: {snapshot.created_at.isoformat()}",
            f"Created by: {snapshot.created_by}",
        ]
    )
    if snapshot.approved_by:
        lines.append(f"Approved by: {snapshot.approved_by}")
    if snapshot.approved_at:
        lines.append(f"Approved at: {snapshot.approved_at.isoformat()}")
    lines.extend(_txt_section("Content", "Final report content is available as Markdown and HTML."))
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
        ":root{color-scheme:light;--ink:#17202a;--muted:#667085;--line:#e5e9f0;--panel:#f8fafc;--badge:#eef2f6}",
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:900px;margin:44px auto;padding:0 24px;line-height:1.62;color:var(--ink);background:#fff}",
        "h1{font-size:34px;line-height:1.08;margin:0 0 18px}h2{font-size:20px;margin:0 0 10px;line-height:1.2}p{margin:0 0 12px}.meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:20px 0 26px;padding:16px;border:1px solid var(--line);border-radius:10px;background:var(--panel);color:var(--muted);font-size:13px}.badge{display:inline-block;border-radius:999px;background:var(--badge);padding:2px 8px;font-size:12px;color:#344054}section{border:1px solid var(--line);border-radius:10px;padding:18px;margin:14px 0;background:#fff}pre{white-space:pre-wrap;background:var(--panel);border-radius:8px;padding:12px;overflow:auto}ul{margin-top:8px}@media print{body{margin:0;max-width:none}.meta,section{break-inside:avoid}}",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{title}</h1>",
        "<div class='meta'>",
        f"<div><strong>Status</strong><br><span class='badge'>{html.escape(report.status.value)}</span></div>",
        f"<div><strong>Approval</strong><br><span class='badge'>{html.escape(report.approval_status.value)}</span></div>",
        f"<div><strong>Template</strong><br>{html.escape(report.template.value)}</div>",
        f"<div><strong>Version</strong><br>{report.version}</div>",
        f"<div><strong>Created</strong><br>{html.escape(report.created_at.isoformat())}</div>",
        f"<div><strong>Updated</strong><br>{html.escape(report.updated_at.isoformat())}</div>",
    ]
    if report.approved_by:
        body.append(f"<div><strong>Approved by</strong><br>{html.escape(report.approved_by)}</div>")
    if report.approved_at:
        body.append(f"<div><strong>Approved at</strong><br>{html.escape(report.approved_at.isoformat())}</div>")
    body.append("</div>")
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


def _append_report_text_pages(pdf: Any, plt: Any, report: ShareableReport) -> None:
    sections = sorted(report.sections, key=lambda item: (item.order, item.title))
    page_lines = [
        report.title,
        "",
        f"Template: {_title_value(report.template.value)}",
        f"Version: {report.version}",
        f"Approval: {_title_value(report.approval_status.value)}",
        f"Updated: {report.updated_at.isoformat()}",
        "",
    ]
    for section in sections:
        page_lines.extend([section.title, ""])
        page_lines.extend(_wrap_pdf_text(section.content or "No content.", width=92))
        page_lines.append("")
    for chunk in _chunk_lines(page_lines, 42):
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0.08, 0.06, 0.84, 0.88])
        ax.axis("off")
        y = 1.0
        for index, line in enumerate(chunk):
            is_title = index == 0 and line == report.title
            is_section = line and line in {section.title for section in sections}
            ax.text(
                0,
                y,
                line,
                fontsize=16 if is_title else 12 if is_section else 9.5,
                fontweight="bold" if is_title or is_section else "normal",
                color="#111827" if is_title or is_section else "#374151",
                va="top",
                wrap=True,
            )
            y -= 0.045 if is_title or is_section else 0.026
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _pdf_artifacts(report: ShareableReport, artifacts: list[Artifact]) -> list[Artifact]:
    by_id = {artifact.artifact_id: artifact for artifact in artifacts}
    selected = [by_id[artifact_id] for artifact_id in report.source_artifact_ids if artifact_id in by_id]
    if selected:
        return [artifact for artifact in selected if artifact.artifact_type in {ArtifactType.CHART, ArtifactType.TABLE}]
    return [
        artifact
        for artifact in artifacts
        if artifact.visibility == ArtifactVisibility.USER and artifact.artifact_type in {ArtifactType.CHART, ArtifactType.TABLE}
    ][:8]


def _chart_artifact_figure(plt: Any, artifact: Artifact) -> Any | None:
    chart = _normalize_chart_artifact(artifact)
    if not chart or (not chart.get("points") and not chart.get("groups")):
        return None
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_title(str(chart["title"]), fontsize=14, fontweight="bold", loc="left", pad=14)
    chart_type = chart["type"]
    points = chart["points"]
    labels = [str(point.get("label") or "") for point in points]
    values = [float(point.get("value") or 0) for point in points]
    if chart_type == "histogram_comparison":
        groups = chart.get("groups") or []
        bin_labels = list(dict.fromkeys(label for group in groups for label in group.get("labels", [])))
        if not groups or not bin_labels:
            plt.close(fig)
            return None
        width = min(0.8 / max(len(groups), 1), 0.38)
        centers = list(range(len(bin_labels)))
        for group_index, group in enumerate(groups):
            offset = (group_index - (len(groups) - 1) / 2) * width
            counts_by_label = dict(zip(group.get("labels", []), group.get("counts", []), strict=False))
            counts = [float(counts_by_label.get(label) or 0) for label in bin_labels]
            ax.bar([center + offset for center in centers], counts, width=width, label=str(group.get("label") or f"Group {group_index + 1}"), alpha=0.82)
        ax.set_xticks(centers)
        ax.set_xticklabels(bin_labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("record count")
        ax.legend(frameon=False, fontsize=8)
    elif chart_type in {"histogram", "bar"}:
        ax.bar(range(len(values)), values, color="#2563eb", alpha=0.88)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(str(chart.get("y_label") or "value"))
    elif chart_type == "line":
        ax.plot(range(len(values)), values, color="#2563eb", linewidth=2.4, marker="o", markersize=3)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(str(chart.get("y_label") or "value"))
    elif chart_type == "scatter":
        xs = [float(point.get("x") or index) for index, point in enumerate(points)]
        ys = [float(point.get("y") or point.get("value") or 0) for point in points]
        ax.scatter(xs, ys, color="#2563eb", alpha=0.72)
        ax.set_xlabel(str(chart.get("x_label") or "x"))
        ax.set_ylabel(str(chart.get("y_label") or "y"))
    else:
        ax.bar(range(len(values)), values, color="#2563eb", alpha=0.88)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def _table_artifact_figure(plt: Any, artifact: Artifact) -> Any | None:
    rows = _artifact_rows(artifact.content)
    if not rows:
        return None
    columns = _table_columns(rows)[:6]
    visible_rows = rows[:12]
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.set_title(artifact.title or "Table", fontsize=14, fontweight="bold", loc="left", pad=12)
    cell_text = [[_short_cell(row.get(column)) for column in columns] for row in visible_rows]
    table = ax.table(cellText=cell_text, colLabels=columns, loc="upper left", cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.3)
    for (row_index, _column_index), cell in table.get_celld().items():
        if row_index == 0:
            cell.set_facecolor("#eff6ff")
            cell.set_text_props(weight="bold", color="#111827")
        else:
            cell.set_facecolor("#ffffff" if row_index % 2 else "#f8fafc")
    fig.tight_layout()
    return fig


def _normalize_chart_artifact(artifact: Artifact) -> dict[str, Any] | None:
    content = artifact.content if isinstance(artifact.content, dict) else {}
    raw_type = str(content.get("chart_type") or content.get("type") or "bar").lower()
    chart_type = "line" if "line" in raw_type else "histogram" if "hist" in raw_type else "scatter" if "scatter" in raw_type else "bar"
    rows = _artifact_rows(content.get("rows"))
    x_key = str(content.get("x") or content.get("x_field") or "label")
    y_key = str(content.get("y") or content.get("y_field") or "value")
    if chart_type == "histogram":
        groups = _histogram_comparison_groups(content)
        if groups:
            points = [
                {"label": f"{group['label']}: {label}", "value": count}
                for group in groups
                for label, count in zip(group["labels"], group["counts"], strict=False)
            ]
            return {
                "type": "histogram_comparison",
                "title": artifact.title or "Histogram comparison",
                "points": points,
                "groups": groups,
                "x_label": x_key,
                "y_label": "record count",
            }
        points = _histogram_points(content, rows)
        return {"type": chart_type, "title": artifact.title or "Histogram", "points": points, "x_label": x_key, "y_label": "count"}
    if chart_type == "scatter":
        points = []
        for index, row in enumerate(rows):
            x = _number(row.get(x_key))
            y = _number(row.get(y_key))
            if x is not None and y is not None:
                points.append({"label": str(row.get("label") or f"Point {index + 1}"), "value": y, "x": x, "y": y})
        return {"type": chart_type, "title": artifact.title or "Scatter plot", "points": points[:80], "x_label": x_key, "y_label": y_key}
    points = []
    for row in rows:
        label = str(row.get(x_key) or row.get("period") or row.get("label") or "")
        value = _number(row.get(y_key)) or _number(row.get("mean")) or _number(row.get("value")) or _number(row.get("count"))
        if label or value is not None:
            points.append({"label": label or "Value", "value": value or 0})
    return {"type": chart_type, "title": artifact.title or "Chart", "points": points[:24], "x_label": x_key, "y_label": y_key}


def _histogram_comparison_groups(content: dict[str, Any]) -> list[dict[str, Any]]:
    raw_groups = content.get("comparison_groups")
    if not isinstance(raw_groups, list):
        return []
    groups: list[dict[str, Any]] = []
    total_count = 0
    for index, raw_group in enumerate(raw_groups):
        if not isinstance(raw_group, dict):
            continue
        label = str(raw_group.get("label") or raw_group.get("group") or f"Group {index + 1}")
        expected_count = (
            _number(raw_group.get("row_count"))
            or _number(raw_group.get("record_count"))
            or _number(raw_group.get("n"))
        )
        bins = _artifact_rows(raw_group.get("bins"))
        labels: list[str] = []
        counts: list[float] = []
        for bin_index, row in enumerate(bins):
            count = _number(row.get("count")) or _number(row.get("value")) or 0
            left = _number(row.get("left"))
            right = _number(row.get("right"))
            bin_label = str(row.get("label") or row.get("bin") or "")
            if not bin_label and left is not None and right is not None:
                bin_label = f"{left:g}-{right:g}"
            labels.append(bin_label or f"Bin {bin_index + 1}")
            counts.append(float(count))
        group_total = sum(counts)
        if not labels or group_total <= 0:
            continue
        if expected_count is not None and int(round(group_total)) != int(round(expected_count)):
            continue
        total_count += int(round(group_total))
        groups.append({"label": label, "labels": labels[:24], "counts": counts[:24]})
    declared_total = _number(content.get("row_count"))
    if declared_total is not None and groups and int(round(declared_total)) != total_count:
        return []
    return groups


def _histogram_points(content: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = content.get("bin_counts")
    edges = content.get("bin_edges")
    if isinstance(counts, list) and isinstance(edges, list) and len(edges) == len(counts) + 1:
        points = []
        for index, count in enumerate(counts):
            left = _number(edges[index])
            right = _number(edges[index + 1])
            label = f"{left:g}-{right:g}" if left is not None and right is not None else f"Bin {index + 1}"
            points.append({"label": label, "value": _number(count) or 0})
        return points[:24]
    bins = content.get("bins")
    source_rows = _artifact_rows(bins) or rows
    points = []
    for index, row in enumerate(source_rows):
        label = str(row.get("label") or row.get("bin") or f"Bin {index + 1}")
        value = _number(row.get("count")) or _number(row.get("value")) or 0
        points.append({"label": label, "value": value})
    return points[:24]


def _artifact_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        nested = value.get("rows") or value.get("content") or value.get("data")
        if isinstance(nested, list):
            return [row for row in nested if isinstance(row, dict)]
    return []


def _table_columns(rows: list[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(str(key))
    return columns


def _short_cell(value: Any) -> str:
    if value is None:
        return ""
    text = f"{value:.4g}" if isinstance(value, float) else str(value)
    return textwrap.shorten(text, width=34, placeholder="...")


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _wrap_pdf_text(value: str, width: int) -> list[str]:
    lines: list[str] = []
    for raw_line in str(value).splitlines() or [""]:
        if not raw_line.strip():
            lines.append("")
            continue
        prefix = "- " if raw_line.strip().startswith("- ") else ""
        wrapped = textwrap.wrap(raw_line.strip(), width=width, subsequent_indent="  " if prefix else "")
        lines.extend(wrapped or [""])
    return lines


def _chunk_lines(lines: list[str], size: int) -> list[list[str]]:
    return [lines[index:index + size] for index in range(0, len(lines), size)] or [[]]


def _finding_markdown(finding: Finding) -> list[str]:
    metadata = finding.metadata or {}
    conclusion = _metadata_text(metadata, "conclusion") or finding.text
    lines = [f"### {finding.title or 'Finding'}", "", f"**Status:** `{finding.status.value}`", "", conclusion, ""]
    structured_rows = [
        ("Confidence", _metadata_text(metadata, "confidence_reason")),
        ("Evidence", _metadata_text(metadata, "evidence_reason")),
        ("Implication", _metadata_text(metadata, "business_implication")),
        ("Limitation", _metadata_text(metadata, "limitation")),
        ("Recommended validation", _metadata_text(metadata, "recommended_validation")),
    ]
    structured_rows = [(label, value) for label, value in structured_rows if value]
    if structured_rows:
        lines.extend(f"**{label}:** {value}" for label, value in structured_rows)
        lines.append("")
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
        return [_summarize_structured_content(content)]
    if artifact.artifact_type == ArtifactType.PYTHON_CODE:
        return ["```python", str(content or ""), "```"]
    if artifact.artifact_type == ArtifactType.SQL:
        return ["```sql", str(content or ""), "```"]
    return [str(content or "")]


def _txt_header(title: str) -> list[str]:
    clean_title = str(title or "Untitled report").strip() or "Untitled report"
    return [clean_title.upper(), "=" * min(max(len(clean_title), 18), 72), ""]


def _txt_section(title: str, content: str) -> list[str]:
    clean_title = str(title or "Section").strip().upper()
    body = str(content or "").strip() or "No content."
    return ["", clean_title, "-" * min(max(len(clean_title), 8), 72), body, ""]


def _title_value(value: str) -> str:
    return str(value or "").replace("_", " ").title()


def _summarize_structured_content(content: Any) -> str:
    if isinstance(content, list):
        if not content:
            return "Empty list."
        if all(isinstance(row, dict) for row in content):
            columns = list(content[0].keys()) if content else []
            return f"Table with {len(content)} row(s)" + (f" and columns: {', '.join(map(str, columns[:8]))}." if columns else ".")
        return f"List with {len(content)} item(s)."
    if isinstance(content, dict):
        keys = list(content.keys())
        return "Structured content with fields: " + ", ".join(map(str, keys[:12])) + "."
    return str(content or "")


def _report_from_snapshot_json(data: dict[str, Any], snapshot: FinalReportSnapshot) -> ShareableReport | None:
    if not isinstance(data, dict) or not data:
        return None
    sections: list[ReportSection] = []
    for index, item in enumerate(data.get("sections") or []):
        if not isinstance(item, dict):
            continue
        sections.append(
            ReportSection(
                section_id=str(item.get("section_id") or item.get("id") or ""),
                title=str(item.get("title") or f"Section {index + 1}"),
                content=str(item.get("content") or ""),
                order=int(item.get("order") or index),
                artifact_ids=list(item.get("artifact_ids") or []),
                metadata=dict(item.get("metadata") or {}),
            )
        )
    try:
        template = ShareableReportTemplate(data.get("template") or "executive_summary")
    except ValueError:
        template = ShareableReportTemplate.EXECUTIVE_SUMMARY
    try:
        status = ShareableReportStatus(data.get("status") or "ready")
    except ValueError:
        status = ShareableReportStatus.READY
    try:
        approval_status = ReportApprovalStatus(data.get("approval_status") or snapshot.approval_status.value)
    except ValueError:
        approval_status = snapshot.approval_status
    return ShareableReport(
        report_id=str(data.get("report_id") or snapshot.report_id),
        investigation_id=str(data.get("investigation_id") or snapshot.investigation_id),
        title=str(data.get("title") or snapshot.title),
        template=template,
        status=status,
        version=int(data.get("version") or snapshot.report_version),
        approval_status=approval_status,
        approved_at=snapshot.approved_at,
        approved_by=snapshot.approved_by,
        sections=sections,
        metadata=dict(data.get("metadata") or {}),
    )


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


def _metadata_text(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key)
    return str(value).strip() if isinstance(value, str) and value.strip() else ""


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
