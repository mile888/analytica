from __future__ import annotations

import pandas as pd
import streamlit as st

from source.product.investigation import DecisionMetadata, DecisionStatus
from source.product.final_report_registry import list_published_reports
from source.product.report_service import ReportEditingService
from source.product.store_factory import create_investigation_store


st.set_page_config(page_title="Analytica · Published Reports", layout="wide")

if "published_reports_store" not in st.session_state:
    st.session_state.published_reports_store = create_investigation_store()

store = st.session_state.published_reports_store
service = ReportEditingService(store)

st.title("Published Reports")
st.caption("Final deliverables published from Investigations")

filter_cols = st.columns([0.34, 0.16, 0.2, 0.15, 0.15])
with filter_cols[0]:
    search = st.text_input("Search", placeholder="Search by report, tag, owner, area, or decision")
with filter_cols[1]:
    status_label = st.selectbox("Status", ["all", "final", "revoked"], index=0)
with filter_cols[2]:
    investigations = store.list_investigations()
    investigation_options = {"All investigations": None}
    investigation_options.update({item.title: item.investigation_id for item in investigations})
    selected_investigation = st.selectbox("Investigation", list(investigation_options.keys()))
with filter_cols[3]:
    decision_status_filter = st.selectbox("Decision", ["all"] + [item.value for item in DecisionStatus], index=0)
with filter_cols[4]:
    tag_filter = st.text_input("Tag")

business_area_filter = st.text_input("Business area filter", placeholder="Optional")

status = None if status_label == "all" else status_label
decision_status = None if decision_status_filter == "all" else decision_status_filter
investigation_id = investigation_options[selected_investigation]
summaries = list_published_reports(
    store,
    status=status,
    investigation_id=investigation_id,
    search=search,
    decision_status=decision_status,
    business_area=business_area_filter or None,
    tag=tag_filter or None,
)

if summaries:
    table = pd.DataFrame(
        [
            {
                "title": item["title"],
                "status": item["status"],
                "created_at": item["created_at"],
                "created_by": item["created_by"],
                "investigation": item["investigation_title"] or item["investigation_id"],
                "report_version": item["report_version"],
                "approval": item["approval_status"],
                "decision": item["decision_status"],
                "business_area": item["business_area"],
                "tags": ", ".join(item["tags"]),
                "ready": item["readiness_is_ready"],
                "blocking": item["readiness_blocking_count"],
                "warnings": item["readiness_warning_count"],
            }
            for item in summaries
        ]
    )
    st.dataframe(table, width="stretch", hide_index=True)
else:
    st.info("No published reports yet.")

for summary in summaries:
    snapshot = store.get_final_report_snapshot(summary["snapshot_id"])
    border = snapshot.status.value == "final"
    with st.container(border=border):
        st.subheader(snapshot.title)
        st.caption(
            f"{snapshot.status.value} · report v{snapshot.report_version} · "
            f"created {snapshot.created_at.isoformat()} by {snapshot.created_by}"
        )
        if snapshot.decision_metadata.short_description:
            st.markdown(snapshot.decision_metadata.short_description)
        if snapshot.decision_metadata.tags:
            st.caption("Tags: " + ", ".join(snapshot.decision_metadata.tags))
        meta_cols = st.columns(4)
        meta_cols[0].metric("Approval", snapshot.approval_status.value)
        meta_cols[1].metric("Decision", snapshot.decision_metadata.decision_status.value)
        meta_cols[2].metric("Ready", str(summary["readiness_is_ready"]))
        meta_cols[3].metric("Warnings", summary["readiness_warning_count"])

        action_cols = st.columns(3)
        with action_cols[0]:
            st.download_button(
                "Download Markdown",
                data=snapshot.markdown_content.encode("utf-8"),
                file_name=f"final_report_{snapshot.snapshot_id}.md",
                mime="text/markdown",
                width="stretch",
                key=f"published-md-{snapshot.snapshot_id}",
            )
        with action_cols[1]:
            st.download_button(
                "Download HTML",
                data=snapshot.html_content.encode("utf-8"),
                file_name=f"final_report_{snapshot.snapshot_id}.html",
                mime="text/html",
                width="stretch",
                key=f"published-html-{snapshot.snapshot_id}",
            )
        with action_cols[2]:
            if st.button(
                "Revoke snapshot",
                disabled=snapshot.status.value == "revoked",
                width="stretch",
                key=f"published-revoke-{snapshot.snapshot_id}",
            ):
                service.revoke_final_report_snapshot(snapshot.snapshot_id, reason="Revoked from Published Reports")
                st.rerun()

        with st.expander("Details", expanded=False):
            st.markdown(f"Snapshot `{snapshot.snapshot_id}`")
            st.markdown(f"Investigation `{snapshot.investigation_id}`")
            if summary["investigation_title"]:
                st.markdown(f"Investigation title: **{summary['investigation_title']}**")
            st.markdown(f"Report `{snapshot.report_id}` · v{snapshot.report_version}")
            if snapshot.approved_by:
                st.markdown(f"Approved by **{snapshot.approved_by}**")
            if snapshot.approved_at:
                st.markdown(f"Approved at `{snapshot.approved_at.isoformat()}`")
            readiness = snapshot.readiness_snapshot or {}
            st.markdown(
                f"Readiness: `{'ready' if readiness.get('is_ready') else 'needs_attention'}` · "
                f"{readiness.get('blocking_count', 0)} blocking · {readiness.get('warning_count', 0)} warnings"
            )
            checks = readiness.get("checks") or []
            if checks:
                st.markdown("Readiness checks")
                for check in checks:
                    st.caption(f"{check.get('status')} · {check.get('message')}")

            st.markdown("Metadata is editable; final report content is immutable.")
            with st.form(f"metadata-{snapshot.snapshot_id}"):
                tags = st.text_input("Tags", value=", ".join(snapshot.decision_metadata.tags))
                owner = st.text_input("Owner", value=snapshot.decision_metadata.owner or "")
                audience = st.text_input("Audience", value=snapshot.decision_metadata.audience or "")
                business_area = st.text_input("Business area", value=snapshot.decision_metadata.business_area or "")
                decision_date = st.text_input("Decision date", value=snapshot.decision_metadata.decision_date or "")
                decision_status = st.selectbox(
                    "Decision status",
                    [item.value for item in DecisionStatus],
                    index=[item.value for item in DecisionStatus].index(snapshot.decision_metadata.decision_status.value),
                    key=f"decision-status-{snapshot.snapshot_id}",
                )
                short_description = st.text_area(
                    "Short description",
                    value=snapshot.decision_metadata.short_description or "",
                    height=90,
                )
                save_metadata = st.form_submit_button("Save metadata", width="stretch")
                if save_metadata:
                    service.update_final_report_metadata(
                        snapshot.snapshot_id,
                        DecisionMetadata(
                            tags=[item for item in tags.split(",")],
                            owner=owner or None,
                            audience=audience or None,
                            business_area=business_area or None,
                            decision_date=decision_date or None,
                            decision_status=DecisionStatus(decision_status),
                            short_description=short_description or None,
                        ),
                    )
                    st.rerun()
