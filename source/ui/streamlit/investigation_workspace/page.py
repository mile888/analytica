from __future__ import annotations

import io
import json
from dataclasses import asdict
from typing import Any

import pandas as pd
import streamlit as st

from source.config import DEFAULT_DATA_PATH
from source.dataframe import read_csv_dataset
from source.product.exporter import (
    export_investigation_html,
    export_investigation_markdown,
    export_final_report_snapshot_txt,
    export_shareable_report_html,
    export_shareable_report_markdown,
    export_shareable_report_txt,
)
from source.product.data_context import build_data_source_usage_context
from source.product.event_stream import build_event_stream_response
from source.product.data_profiling import profile_dataframe
from source.product.question_suggestions import build_data_aware_question_suggestions
from source.product.report_builder import build_shareable_report
from source.product.report_service import ReportEditingService
from source.product.readiness import evaluate_report_readiness
from source.product import (
    ArtifactType,
    ArtifactVisibility,
    FindingStatus,
    Investigation,
    InvestigationMessage,
    InvestigationMessageRole,
    InvestigationMessageType,
    InvestigationRunService,
    InvestigationService,
    InvestigationStatus,
    ReportCommentStatus,
    SectionReviewStatus,
    ShareableReport,
    ShareableReportTemplate,
    create_investigation_store,
)


WORKSPACE_PRODUCT_VERSION = "investigation-workspace-v5"


def load_default_df() -> pd.DataFrame | None:
    if DEFAULT_DATA_PATH.exists():
        try:
            return read_csv_dataset(DEFAULT_DATA_PATH)
        except Exception:
            return None
    return None


def load_uploaded_csv(uploaded_file, sep: str, encoding: str) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    return pd.read_csv(io.BytesIO(raw), sep=sep, encoding=encoding)


def init_workspace_state() -> None:
    if (
        "investigation_store" not in st.session_state
        or
        "investigation_service" not in st.session_state
        or st.session_state.get("investigation_workspace_version") != WORKSPACE_PRODUCT_VERSION
    ):
        st.session_state.investigation_store = create_investigation_store()
        st.session_state.investigation_service = InvestigationService(st.session_state.investigation_store)
        st.session_state.investigation_run_service = InvestigationRunService(st.session_state.investigation_store)
        st.session_state.investigation_workspace_version = WORKSPACE_PRODUCT_VERSION
    if "workspace_df" not in st.session_state:
        existing_df = st.session_state.get("df")
        st.session_state.workspace_df = existing_df if existing_df is not None else load_default_df()
    if "selected_investigation_id" not in st.session_state:
        st.session_state.selected_investigation_id = None


def status_badge(status: str) -> None:
    st.markdown(f"`{status}`")


def run_stage_label(stage: str) -> str:
    return {
        "preparing_data": "Preparing data",
        "building_context": "Building context",
        "running_analysis": "Running analysis",
        "validating_results": "Validating results",
        "generating_report": "Generating report",
        "completed": "Completed",
    }.get(stage, str(stage).replace("_", " ").title())


def artifact_label(artifact_type: ArtifactType | str) -> str:
    value = str(artifact_type)
    if "." in value:
        value = value.rsplit(".", 1)[-1]
    return value


def render_artifact_content(artifact_type: ArtifactType | str, content: Any) -> None:
    type_value = artifact_label(artifact_type)
    if type_value == ArtifactType.PYTHON_CODE.value:
        st.code(str(content or ""), language="python")
    elif type_value == ArtifactType.SQL.value:
        st.code(str(content or ""), language="sql")
    elif type_value == ArtifactType.TABLE.value:
        if isinstance(content, list) and all(isinstance(row, dict) for row in content):
            st.dataframe(pd.DataFrame(content), width="stretch", height=260)
        elif isinstance(content, dict):
            st.dataframe(pd.DataFrame([content]), width="stretch", height=120)
        else:
            st.code(str(content or ""), language="text")
    elif type_value in {ArtifactType.REPORT.value, ArtifactType.TEXT.value}:
        st.markdown(str(content or ""))
    elif type_value == ArtifactType.VALIDATION.value:
        if isinstance(content, (list, dict)):
            st.json(content)
        else:
            st.code(str(content or ""), language="text")
    elif isinstance(content, (list, dict)):
        st.json(content)
    else:
        st.code(str(content or ""), language="text")


def json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def render_report(report) -> None:
    if report.answer:
        st.markdown("#### Answer")
        st.markdown(report.answer)
    if report.key_findings:
        st.markdown("#### Key findings")
        for item in report.key_findings:
            st.markdown(f"- {item}")
    if report.evidence:
        st.markdown("#### Evidence")
        for item in report.evidence:
            st.markdown(f"- {item}")
    if report.limitations:
        st.markdown("#### Limitations")
        for item in report.limitations:
            st.markdown(f"- {item}")
    if report.next_steps:
        st.markdown("#### Next steps")
        for item in report.next_steps:
            st.markdown(f"- {item}")


def render_shareable_report_preview(report: ShareableReport) -> None:
    st.markdown(f"### {report.title}")
    st.caption(
        f"{report.template.value} · {report.status.value} · {report.approval_status.value} · v{report.version}"
    )
    if report.reviewer_notes:
        st.info(report.reviewer_notes)
    for section in sorted(report.sections, key=lambda item: (item.order, item.title)):
        st.markdown(f"#### {section.title}")
        st.markdown(section.content or "_No content._")


def render_shareable_report_editor(report: ShareableReport, store) -> None:
    editor = ReportEditingService(store)
    st.markdown("#### Edit report")
    sections = sorted(report.sections, key=lambda item: (item.order, item.title))
    for idx, section in enumerate(sections):
        section_comments = store.list_report_comments(report.report_id, section.section_id)
        open_count = sum(1 for comment in section_comments if comment.status == ReportCommentStatus.OPEN)
        comment_label = f" · {open_count} open comments" if open_count else ""
        with st.expander(
            f"{section.order}. {section.title} · {section.review_status.value} · v{section.version}{comment_label}",
            expanded=False,
        ):
            status_cols = st.columns(2)
            with status_cols[0]:
                if st.button(
                    "Approve section",
                    key=f"section-approve-{report.report_id}-{section.section_id}",
                    disabled=section.review_status == SectionReviewStatus.APPROVED,
                    width="stretch",
                ):
                    updated = editor.approve_section(report.report_id, section.section_id)
                    st.session_state.selected_shareable_report_id = updated.report_id
                    st.rerun()
            with status_cols[1]:
                if st.button(
                    "Request section changes",
                    key=f"section-changes-{report.report_id}-{section.section_id}",
                    disabled=section.review_status == SectionReviewStatus.CHANGES_REQUESTED,
                    width="stretch",
                ):
                    updated = editor.request_section_changes(report.report_id, section.section_id)
                    st.session_state.selected_shareable_report_id = updated.report_id
                    st.rerun()
            with st.form(f"edit-section-{report.report_id}-{section.section_id}"):
                new_title = st.text_input("Section title", value=section.title)
                new_content = st.text_area("Section content", value=section.content, height=180)
                save = st.form_submit_button("Save section changes", width="stretch")
                if save:
                    updated = report
                    if new_title != section.title:
                        updated = editor.update_section_title(report.report_id, section.section_id, new_title)
                    if new_content != section.content:
                        updated = editor.update_section_content(updated.report_id, section.section_id, new_content)
                    st.session_state.selected_shareable_report_id = updated.report_id
                    st.rerun()

            move_cols = st.columns(4)
            with move_cols[0]:
                if st.button("Move up", key=f"up-{report.report_id}-{section.section_id}", disabled=idx == 0):
                    ids = [item.section_id for item in sections]
                    ids[idx - 1], ids[idx] = ids[idx], ids[idx - 1]
                    updated = editor.reorder_sections(report.report_id, ids)
                    st.session_state.selected_shareable_report_id = updated.report_id
                    st.rerun()
            with move_cols[1]:
                if st.button(
                    "Move down",
                    key=f"down-{report.report_id}-{section.section_id}",
                    disabled=idx == len(sections) - 1,
                ):
                    ids = [item.section_id for item in sections]
                    ids[idx + 1], ids[idx] = ids[idx], ids[idx + 1]
                    updated = editor.reorder_sections(report.report_id, ids)
                    st.session_state.selected_shareable_report_id = updated.report_id
                    st.rerun()
            with move_cols[2]:
                if st.button("Duplicate", key=f"dup-{report.report_id}-{section.section_id}"):
                    updated = editor.duplicate_section(report.report_id, section.section_id)
                    st.session_state.selected_shareable_report_id = updated.report_id
                    st.rerun()
            with move_cols[3]:
                if st.button("Delete", key=f"del-{report.report_id}-{section.section_id}"):
                    updated = editor.remove_section(report.report_id, section.section_id)
                    st.session_state.selected_shareable_report_id = updated.report_id
                    st.rerun()

            st.markdown(f"##### Review comments{comment_label}")
            if section_comments:
                for comment in section_comments:
                    with st.container(border=True):
                        st.markdown(f"`{comment.status.value}` · {comment.author}")
                        st.markdown(comment.text)
                        if comment.status == ReportCommentStatus.OPEN and st.button(
                            "Resolve comment",
                            key=f"resolve-{report.report_id}-{comment.comment_id}",
                            width="stretch",
                        ):
                            editor.resolve_comment(comment.comment_id)
                            st.rerun()
            else:
                st.caption("No comments on this section.")

            with st.form(f"comment-section-{report.report_id}-{section.section_id}"):
                comment_text = st.text_area("Add review comment", height=90)
                add_comment = st.form_submit_button("Add comment", width="stretch")
                if add_comment and comment_text.strip():
                    editor.add_comment(
                        report.report_id,
                        section.section_id,
                        comment_text.strip(),
                        author="reviewer",
                    )
                    st.rerun()

    with st.form(f"add-section-{report.report_id}"):
        st.markdown("##### Add section")
        title = st.text_input("New section title")
        content = st.text_area("New section content", height=120)
        add = st.form_submit_button("Add section", width="stretch")
        if add and title.strip():
            updated = editor.add_section(report.report_id, title=title.strip(), content=content)
            st.session_state.selected_shareable_report_id = updated.report_id
            st.rerun()


def infer_investigation_suggestions(
    df: pd.DataFrame | None,
    usage_context: Any | None = None,
) -> list[str]:
    profile = None if usage_context or df is None or df.empty else profile_dataframe(df)
    return build_data_aware_question_suggestions(profile=profile, usage_context=usage_context, limit=6)


def render_investigation(investigation: Investigation) -> None:
    service: InvestigationService = st.session_state.investigation_service
    store = service.store

    header_cols = st.columns([0.72, 0.28], vertical_alignment="center")
    with header_cols[0]:
        st.title(investigation.title)
        st.caption(f"Investigation `{investigation.investigation_id}`")
    with header_cols[1]:
        st.metric("Status", investigation.status.value)

    st.subheader("Question")
    st.markdown(investigation.user_question)

    with st.expander("Conversation and follow-ups", expanded=False):
        messages = store.list_investigation_messages(investigation.investigation_id)
        if messages:
            for message in messages:
                st.markdown(f"**{message.role.value} · {message.message_type.value}**")
                st.caption(message.created_at.isoformat())
                st.markdown(message.content)
                st.divider()
        else:
            st.caption("No follow-up messages yet.")

        follow_up = st.text_area(
            "Analyze",
            key=f"follow-up-{investigation.investigation_id}",
            placeholder="Analyze another question...",
            height=90,
        )
        if st.button("Analyze", key=f"ask-follow-up-{investigation.investigation_id}", width="stretch"):
            if follow_up.strip():
                message = store.add_investigation_message(
                    InvestigationMessage(
                        investigation_id=investigation.investigation_id,
                        role=InvestigationMessageRole.USER,
                        message_type=InvestigationMessageType.FOLLOW_UP,
                        content=follow_up.strip(),
                    )
                )
                run_service: InvestigationRunService = st.session_state.investigation_run_service
                product_run = run_service.run_investigation(
                    investigation.investigation_id,
                    data_source_ids=[st.session_state.selected_data_source_id]
                    if st.session_state.get("selected_data_source_id")
                    else [],
                    df=st.session_state.workspace_df,
                    message_id=message.message_id,
                )
                if product_run.status.value == "failed":
                    st.error(product_run.error_message or "Follow-up run failed.")
                st.rerun()
            else:
                st.warning("Write a follow-up question first.")

    meta_cols = st.columns(4)
    meta_cols[0].metric("Runs", len(investigation.runs))
    meta_cols[1].metric("Artifacts", len(investigation.artifacts))
    meta_cols[2].metric("Findings", len(investigation.findings))
    meta_cols[3].metric("Data sources", len(investigation.linked_data_source_ids or investigation.data_sources))

    if investigation.linked_data_source_ids:
        st.caption("Linked data sources: " + ", ".join(investigation.linked_data_source_ids))
        with st.expander("Linked data source context", expanded=False):
            for data_source_id in investigation.linked_data_source_ids:
                try:
                    context = build_data_source_usage_context(store, data_source_id)
                except KeyError:
                    st.warning(f"Data source not found: {data_source_id}")
                    continue
                schema = context.schema_summary
                st.markdown(f"**{context.name}** · `{context.status.value}`")
                st.caption(
                    f"{schema.get('row_count', 'unknown')} rows · "
                    f"{schema.get('column_count', 'unknown')} columns"
                )
                roles = schema.get("roles", {})
                if roles:
                    role_preview = []
                    for role, names in roles.items():
                        if names:
                            role_preview.append(f"{role}: {', '.join(names[:5])}")
                    if role_preview:
                        st.caption("; ".join(role_preview))
                if context.caveats:
                    for caveat in context.caveats[:4]:
                        st.warning(caveat)

    st.divider()

    left, right = st.columns([0.62, 0.38], gap="large")

    with left:
        st.subheader("Findings")
        if investigation.findings:
            for finding in investigation.findings:
                with st.container(border=True):
                    status_label = finding.status.value
                    st.markdown(f"**{finding.title or 'Finding'}** · `{status_label}`")
                    st.markdown(finding.text)
                    if finding.evidence:
                        st.caption("Evidence: " + "; ".join(finding.evidence))
                    action_cols = st.columns(2)
                    with action_cols[0]:
                        if st.button(
                            "Accept finding",
                            key=f"accept-{investigation.investigation_id}-{finding.finding_id}",
                            disabled=finding.status == FindingStatus.ACCEPTED,
                            width="stretch",
                        ):
                            store.update_finding_status(
                                investigation.investigation_id,
                                finding.finding_id,
                                FindingStatus.ACCEPTED,
                            )
                            st.rerun()
                    with action_cols[1]:
                        if st.button(
                            "Reject finding",
                            key=f"reject-{investigation.investigation_id}-{finding.finding_id}",
                            disabled=finding.status == FindingStatus.REJECTED,
                            width="stretch",
                        ):
                            store.update_finding_status(
                                investigation.investigation_id,
                                finding.finding_id,
                                FindingStatus.REJECTED,
                            )
                            st.rerun()
        else:
            st.info("No findings yet. Run this investigation to populate the workspace.")

        st.subheader("Report")
        if investigation.report:
            render_report(investigation.report)
        else:
            st.info("No DecisionReport yet.")

        st.subheader("Artifacts")
        show_technical = st.toggle("Show technical artifacts", value=False)
        visible_artifacts = [
            artifact
            for artifact in investigation.artifacts
            if artifact.visibility == ArtifactVisibility.USER
            or artifact.pinned
            or (show_technical and artifact.visibility == ArtifactVisibility.TECHNICAL)
        ]
        if visible_artifacts:
            for artifact in visible_artifacts:
                pinned_label = " · pinned" if artifact.pinned else ""
                visibility_label = artifact.visibility.value
                with st.expander(
                    f"{artifact.title or 'Artifact'} · {artifact_label(artifact.artifact_type)} · {visibility_label}{pinned_label}"
                ):
                    render_artifact_content(artifact.artifact_type, artifact.content)
                    pin_label = "Unpin artifact" if artifact.pinned else "Pin artifact"
                    if st.button(
                        pin_label,
                        key=f"pin-{investigation.investigation_id}-{artifact.artifact_id}",
                        width="stretch",
                    ):
                        store.set_artifact_pinned(
                            investigation.investigation_id,
                            artifact.artifact_id,
                            not artifact.pinned,
                        )
                        st.rerun()
                    if artifact.path:
                        st.caption(f"Path: {artifact.path}")
                    if artifact.metadata:
                        st.caption("Metadata")
                        st.json(artifact.metadata)
        else:
            st.info("No artifacts yet.")

        st.subheader("Reports")
        template_options = {
            "Executive Summary": ShareableReportTemplate.EXECUTIVE_SUMMARY,
            "Product Decision Memo": ShareableReportTemplate.PRODUCT_DECISION_MEMO,
            "Technical Appendix": ShareableReportTemplate.TECHNICAL_APPENDIX,
        }
        selected_template_label = st.selectbox(
            "Template",
            list(template_options.keys()),
            key=f"share-template-{investigation.investigation_id}",
        )
        include_report_technical = st.checkbox(
            "Include technical appendix",
            value=False,
            key=f"share-technical-{investigation.investigation_id}",
        )
        if st.button("Create shareable report", key=f"create-share-{investigation.investigation_id}", width="stretch"):
            report = build_shareable_report(
                investigation,
                template=template_options[selected_template_label],
                include_technical=include_report_technical,
            )
            store.create_shareable_report(report)
            st.session_state.selected_shareable_report_id = report.report_id
            st.rerun()

        shareable_reports = store.list_shareable_reports(investigation.investigation_id)
        if shareable_reports:
            report_options = {f"{report.title} · {report.status.value} · v{report.version}": report for report in shareable_reports}
            labels = list(report_options.keys())
            current_report_id = st.session_state.get("selected_shareable_report_id")
            default_index = 0
            for idx, report in enumerate(report_options.values()):
                if report.report_id == current_report_id:
                    default_index = idx
                    break
            selected_report_label = st.selectbox(
                "Existing reports",
                labels,
                index=default_index,
                key=f"share-select-{investigation.investigation_id}",
            )
            selected_report = report_options[selected_report_label]
            st.session_state.selected_shareable_report_id = selected_report.report_id
            with st.container(border=True):
                report_reviewer = ReportEditingService(store)
                st.markdown("#### Report review")
                report_comments = store.list_report_comments(selected_report.report_id)
                readiness = evaluate_report_readiness(
                    selected_report,
                    investigation=investigation,
                    comments=report_comments,
                )
                with st.container(border=True):
                    st.markdown("##### Readiness")
                    if readiness.is_ready:
                        st.success("Ready")
                    else:
                        st.warning("Needs attention")
                    st.caption(f"Generated: {readiness.generated_at.isoformat()}")
                    failed_checks = [check for check in readiness.checks if check.blocking and check.status.value == "failed"]
                    warning_checks = [check for check in readiness.checks if check.status.value == "warning"]
                    if failed_checks:
                        st.markdown("Blocking checks")
                        for check in failed_checks:
                            st.error(check.message)
                    if warning_checks:
                        st.markdown("Warnings")
                        for check in warning_checks:
                            st.warning(check.message)
                    if st.button("Refresh readiness", key=f"refresh-readiness-{selected_report.report_id}"):
                        st.rerun()

                status_cols = st.columns([0.28, 0.24, 0.24, 0.24])
                status_cols[0].markdown(f"`{selected_report.approval_status.value}`")
                with status_cols[1]:
                    if st.button("Send to review", key=f"review-send-{selected_report.report_id}", width="stretch"):
                        selected_report = report_reviewer.send_to_review(selected_report.report_id)
                        st.session_state.selected_shareable_report_id = selected_report.report_id
                        st.rerun()
                reviewer_notes = st.text_area(
                    "Reviewer notes",
                    value=selected_report.reviewer_notes,
                    height=90,
                    key=f"review-notes-{selected_report.report_id}",
                )
                with status_cols[2]:
                    if st.button("Request changes", key=f"review-request-{selected_report.report_id}", width="stretch"):
                        selected_report = report_reviewer.request_changes(selected_report.report_id, reviewer_notes)
                        st.session_state.selected_shareable_report_id = selected_report.report_id
                        st.rerun()
                with status_cols[3]:
                    if st.button("Approve report", key=f"review-approve-{selected_report.report_id}", width="stretch"):
                        try:
                            selected_report = report_reviewer.approve_report(selected_report.report_id)
                            st.session_state.selected_shareable_report_id = selected_report.report_id
                            st.rerun()
                        except ValueError as exc:
                            st.warning(str(exc))

                render_shareable_report_preview(selected_report)
                render_shareable_report_editor(selected_report, store)
                versions = store.list_report_versions(selected_report.report_id)
                if versions:
                    version_labels = {
                        f"v{version.version} · {version.status.value} · {version.version_note or 'original'}": version
                        for version in versions
                    }
                    selected_version_label = st.selectbox(
                        "Report versions",
                        list(version_labels.keys()),
                        index=max(0, len(version_labels) - 1),
                        key=f"versions-{selected_report.report_id}",
                    )
                    selected_version = version_labels[selected_version_label]
                    if st.button(
                        "Restore selected version",
                        key=f"restore-{selected_report.report_id}",
                        disabled=selected_version.is_latest,
                        width="stretch",
                    ):
                        restored = store.restore_report_version(selected_version.report_id)
                        st.session_state.selected_shareable_report_id = restored.report_id
                        st.rerun()
                include_review_comments = st.checkbox(
                    "Include review comments in export",
                    value=False,
                    key=f"share-comments-{selected_report.report_id}",
                )
                export_comments = store.list_report_comments(selected_report.report_id) if include_review_comments else []
                if not readiness.is_ready:
                    st.warning("Final export should be used only after readiness checks pass.")
                st.download_button(
                    "Draft download TXT",
                    data=export_shareable_report_txt(selected_report).encode("utf-8"),
                    file_name=f"shareable_report_{selected_report.report_id}.txt",
                    mime="text/plain",
                    width="stretch",
                    key=f"share-txt-{selected_report.report_id}",
                )
                st.download_button(
                    "Draft download Markdown",
                    data=export_shareable_report_markdown(
                        selected_report,
                        include_comments=include_review_comments,
                        comments=export_comments,
                    ).encode("utf-8"),
                    file_name=f"shareable_report_{selected_report.report_id}.md",
                    mime="text/markdown",
                    width="stretch",
                    key=f"share-md-{selected_report.report_id}",
                )
                st.download_button(
                    "Draft download HTML",
                    data=export_shareable_report_html(
                        selected_report,
                        include_comments=include_review_comments,
                        comments=export_comments,
                    ).encode("utf-8"),
                    file_name=f"shareable_report_{selected_report.report_id}.html",
                    mime="text/html",
                    width="stretch",
                    key=f"share-html-{selected_report.report_id}",
                )
                st.markdown("#### Final report")
                st.caption(
                    f"Approval: `{selected_report.approval_status.value}` · "
                    f"Readiness: `{'ready' if readiness.is_ready else 'needs_attention'}`"
                )
                if selected_report.approval_status.value != "approved":
                    st.warning("Final snapshot normally requires an approved report.")
                if not readiness.is_ready:
                    st.warning("Final snapshot normally requires readiness checks to pass.")
                force_final = st.checkbox(
                    "Force final export despite readiness issues",
                    value=False,
                    key=f"force-final-{selected_report.report_id}",
                )
                if st.button("Create final snapshot", key=f"finalize-{selected_report.report_id}", width="stretch"):
                    try:
                        snapshot = report_reviewer.create_final_report_snapshot(
                            selected_report.report_id,
                            created_by="user",
                            force=force_final,
                        )
                        st.session_state.selected_final_snapshot_id = snapshot.snapshot_id
                        st.rerun()
                    except ValueError as exc:
                        st.warning(str(exc))

                final_snapshots = store.list_final_report_snapshots(report_id=selected_report.report_id)
                if final_snapshots:
                    snapshot_options = {
                        f"{snapshot.title} · v{snapshot.report_version} · {snapshot.status.value} · {snapshot.created_at.isoformat()}": snapshot
                        for snapshot in final_snapshots
                    }
                    selected_snapshot_label = st.selectbox(
                        "Final snapshots",
                        list(snapshot_options.keys()),
                        key=f"final-snapshot-select-{selected_report.report_id}",
                    )
                    selected_snapshot = snapshot_options[selected_snapshot_label]
                    st.caption(
                        f"Created by {selected_snapshot.created_by} · "
                        f"Readiness: `{selected_snapshot.readiness_snapshot.get('is_ready')}`"
                    )
                    st.download_button(
                        "Final deliverable TXT",
                        data=export_final_report_snapshot_txt(selected_snapshot).encode("utf-8"),
                        file_name=f"final_report_{selected_snapshot.snapshot_id}.txt",
                        mime="text/plain",
                        width="stretch",
                        key=f"final-txt-{selected_snapshot.snapshot_id}",
                    )
                    st.download_button(
                        "Final deliverable Markdown",
                        data=selected_snapshot.markdown_content.encode("utf-8"),
                        file_name=f"final_report_{selected_snapshot.snapshot_id}.md",
                        mime="text/markdown",
                        width="stretch",
                        key=f"final-md-{selected_snapshot.snapshot_id}",
                    )
                    st.download_button(
                        "Final deliverable HTML",
                        data=selected_snapshot.html_content.encode("utf-8"),
                        file_name=f"final_report_{selected_snapshot.snapshot_id}.html",
                        mime="text/html",
                        width="stretch",
                        key=f"final-html-{selected_snapshot.snapshot_id}",
                    )
                    if st.button(
                        "Revoke snapshot",
                        key=f"revoke-final-{selected_snapshot.snapshot_id}",
                        disabled=selected_snapshot.status.value == "revoked",
                        width="stretch",
                    ):
                        report_reviewer.revoke_final_report_snapshot(selected_snapshot.snapshot_id, reason="Revoked in workspace")
                        st.rerun()
                else:
                    st.info("No final snapshots yet.")
        else:
            st.info("No shareable reports yet.")

    with right:
        st.subheader("Review")
        status_badge(investigation.status.value)
        if st.button("Mark verified", disabled=investigation.status == InvestigationStatus.VERIFIED, width="stretch"):
            store.update_status(investigation.investigation_id, InvestigationStatus.VERIFIED)
            st.rerun()
        if st.button("Needs more analysis", width="stretch"):
            store.update_status(investigation.investigation_id, InvestigationStatus.NEEDS_MORE_ANALYSIS)
            st.rerun()
        if st.button("Mark failed", disabled=investigation.status == InvestigationStatus.FAILED, width="stretch"):
            store.update_status(investigation.investigation_id, InvestigationStatus.FAILED)
            st.rerun()
        if st.button("Archive", disabled=investigation.status == InvestigationStatus.ARCHIVED, width="stretch"):
            store.update_status(investigation.investigation_id, InvestigationStatus.ARCHIVED)
            st.rerun()

        st.subheader("Export")
        include_technical_export = st.checkbox("Include technical appendix", value=False)
        markdown_report = export_investigation_markdown(
            investigation,
            include_technical=include_technical_export,
        )
        html_report = export_investigation_html(
            investigation,
            include_technical=include_technical_export,
        )
        st.download_button(
            "Download Markdown report",
            data=markdown_report.encode("utf-8"),
            file_name=f"investigation_{investigation.investigation_id}.md",
            mime="text/markdown",
            width="stretch",
        )
        st.download_button(
            "Download HTML report",
            data=html_report.encode("utf-8"),
            file_name=f"investigation_{investigation.investigation_id}.html",
            mime="text/html",
            width="stretch",
        )

        st.subheader("RunTrace")
        if investigation.trace:
            st.dataframe(pd.DataFrame(investigation.trace), width="stretch", height=260)
        elif investigation.runs:
            latest_run = investigation.runs[-1]
            if latest_run.error:
                st.error(latest_run.error)
            else:
                st.info("No trace events captured for the latest run.")
        else:
            st.info("No runs yet.")

        st.subheader("Execution history")
        if investigation.runs:
            for run in reversed(investigation.runs):
                with st.expander(f"{run.run_id} · {run.status.value}"):
                    st.caption(f"Started: {run.started_at.isoformat()}")
                    if run.finished_at:
                        st.caption(f"Finished: {run.finished_at.isoformat()}")
                    if run.error:
                        st.error(run.error)
                    if run.output:
                        st.json(run.output)
        else:
            st.info("No execution history yet.")

        st.subheader("Run History")
        product_runs = store.list_runs_for_investigation(investigation.investigation_id)
        if product_runs:
            for product_run in product_runs[:8]:
                with st.expander(f"{product_run.run_id} · {product_run.status.value} · {product_run.current_stage.value}"):
                    st.caption(f"Created: {product_run.created_at.isoformat()}")
                    if product_run.started_at:
                        st.caption(f"Started: {product_run.started_at.isoformat()}")
                    if product_run.completed_at:
                        st.caption(f"Completed: {product_run.completed_at.isoformat()}")
                    if product_run.data_source_ids:
                        st.caption("Data sources: " + ", ".join(product_run.data_source_ids))
                    if product_run.artifact_ids:
                        st.caption("Artifacts: " + ", ".join(product_run.artifact_ids))
                    if product_run.report_ids:
                        st.caption("Reports: " + ", ".join(product_run.report_ids))
                    if product_run.error_message:
                        st.error(product_run.error_message)
                    if product_run.run_context_summary:
                        st.json(product_run.run_context_summary)
                    events = store.list_investigation_run_events(product_run.run_id)
                    if events:
                        st.markdown("Timeline")
                        timeline = build_event_stream_response(events, cursor=None, limit=100)
                        for event in timeline["events"]:
                            stage = run_stage_label(event.stage.value) if event.stage else ""
                            st.caption(
                                f"{event.created_at.isoformat()} · {event.severity.value} · "
                                f"{stage} · {event.message}"
                            )
                        if timeline["has_more"]:
                            st.caption("More timeline events are available through the API.")
        else:
            st.info("No analyses yet.")

def render_page() -> None:
    st.set_page_config(page_title="Analytica · Investigation Workspace", layout="wide")
    init_workspace_state()

    service: InvestigationService = st.session_state.investigation_service
    store = st.session_state.investigation_store
    if "investigation_run_service" not in st.session_state:
        st.session_state.investigation_run_service = InvestigationRunService(store)
    run_service: InvestigationRunService = st.session_state.investigation_run_service

    with st.sidebar:
        st.header("Workspace")

        st.subheader("Data source")
        data_sources = store.list_data_sources(status="active")
        source_options = {"Workspace dataframe": None}
        source_options.update({source.name: source.data_source_id for source in data_sources})
        selected_source_label = st.selectbox("Existing source", list(source_options.keys()))
        st.session_state.selected_data_source_id = source_options[selected_source_label]
        st.session_state.selected_usage_context = None
        if st.session_state.selected_data_source_id:
            try:
                context = build_data_source_usage_context(store, st.session_state.selected_data_source_id)
                st.session_state.selected_usage_context = context
                schema = context.schema_summary
                st.caption(f"Profile: {schema.get('row_count')} rows · {schema.get('column_count')} columns")
                role_columns = schema.get("roles", {})
                key_columns = []
                for role in ["timestamp", "metric", "dimension", "identifier"]:
                    key_columns.extend(role_columns.get(role, [])[:3])
                if key_columns:
                    st.caption("Key columns: " + ", ".join(key_columns[:8]))
                if context.description:
                    st.caption(context.description.splitlines()[0][:180])
                overridden = [
                    column.name
                    for column in context.column_summaries
                    if any("overrides deterministic role" in note for note in column.notes)
                ]
                if overridden:
                    st.caption(f"Semantic overrides: {len(overridden)} columns")
                if context.caveats:
                    st.warning(context.caveats[0])
                if context.previous_questions:
                    st.caption(f"Previous investigations: {len(context.previous_questions)}")
            except KeyError:
                st.caption("No profile saved for selected source")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        sep = st.selectbox("Separator", [",", ";", "\t"], index=0)
        encoding = st.selectbox("Encoding", ["utf-8", "utf-8-sig", "cp1251"], index=0)
        if uploaded is not None:
            try:
                st.session_state.workspace_df = load_uploaded_csv(uploaded, sep=sep, encoding=encoding)
                st.success("CSV loaded")
            except Exception as exc:
                st.error(f"Could not read CSV: {exc}")

        if st.session_state.workspace_df is not None:
            df: pd.DataFrame = st.session_state.workspace_df
            st.caption(f"Rows: {df.shape[0]} · Columns: {df.shape[1]}")
        else:
            st.warning("No dataset loaded")

        st.divider()
        st.subheader("New Investigation")
        with st.form("new_investigation_form", clear_on_submit=True):
            suggestions = infer_investigation_suggestions(
                st.session_state.workspace_df,
                usage_context=st.session_state.get("selected_usage_context"),
            )
            placeholder = suggestions[0] if suggestions else "What changed in this dataset?"
            question = st.text_area("Question", placeholder=placeholder, height=90)
            title = st.text_input("Title", placeholder="Optional")
            submitted = st.form_submit_button("Create investigation", width="stretch")
            if submitted:
                if question.strip():
                    investigation = service.create_investigation(question=question, title=title or None)
                    if st.session_state.get("selected_data_source_id"):
                        store.link_data_source_to_investigation(
                            investigation.investigation_id,
                            st.session_state.selected_data_source_id,
                        )
                    st.session_state.selected_investigation_id = investigation.investigation_id
                    st.rerun()
                else:
                    st.warning("Add a question before creating an Investigation.")

        suggestions = infer_investigation_suggestions(
            st.session_state.workspace_df,
            usage_context=st.session_state.get("selected_usage_context"),
        )
        if suggestions:
            with st.expander("Suggested investigations", expanded=True):
                for item in suggestions:
                    st.caption(item)

        st.divider()
        st.subheader("Investigations")
        investigations = store.list_investigations()
        if investigations:
            for investigation in investigations:
                label = f"{investigation.title} · {investigation.status.value}"
                if st.button(label, key=f"select-{investigation.investigation_id}", width="stretch"):
                    st.session_state.selected_investigation_id = investigation.investigation_id
                    st.rerun()
        else:
            st.caption("No investigations yet")


    st.title("Investigation Workspace")
    st.caption("Question → Investigation → Artifacts → DecisionReport → Review")

    selected_id = st.session_state.selected_investigation_id
    if not selected_id and store.list_investigations():
        selected_id = store.list_investigations()[0].investigation_id
        st.session_state.selected_investigation_id = selected_id

    if not selected_id:
        st.info("Create an Investigation from the sidebar to begin.")
        if st.session_state.workspace_df is not None:
            with st.expander("Dataset preview", expanded=True):
                st.dataframe(st.session_state.workspace_df.head(50), width="stretch", height=360)
        st.stop()

    try:
        current = store.get_investigation(selected_id)
    except KeyError:
        st.session_state.selected_investigation_id = None
        st.warning("Selected Investigation was not found.")
        st.stop()

    action_cols = st.columns([0.22, 0.78], vertical_alignment="center")
    with action_cols[0]:
        run_disabled = st.session_state.workspace_df is None or current.status.value == "running"
        if st.button("Run investigation", disabled=run_disabled, type="primary", width="stretch"):
            if st.session_state.workspace_df is not None:
                store.replace_data_sources(current.investigation_id, ["workspace_dataframe"])
            with st.status("Running Investigation...", expanded=True) as status:
                product_run = run_service.run_investigation(
                    current.investigation_id,
                    data_source_ids=[st.session_state.selected_data_source_id]
                    if st.session_state.get("selected_data_source_id")
                    else [],
                    df=st.session_state.workspace_df,
                )
                st.write(f"Stage: `{product_run.current_stage.value}`")
                st.write(f"Status: `{product_run.status.value}`")
                if product_run.run_context_summary:
                    st.json(product_run.run_context_summary)
                if product_run.status.value == "failed":
                    status.update(label="Investigation failed", state="error")
                else:
                    status.update(label="Investigation needs review", state="complete")
            st.rerun()

    with action_cols[1]:
        if run_disabled and st.session_state.workspace_df is None:
            st.warning("Load a dataset before running this Investigation.")

    render_investigation(current)

    with st.expander("Raw Investigation object"):
        st.json(json_safe(asdict(current)))
