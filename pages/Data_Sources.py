from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from source.product.data_context import build_data_source_usage_context
from source.product.data_profiling import profile_dataframe
from source.product.data_sources import ColumnSemanticNote, ColumnSemanticRole, DataSource, DataSourceSemanticNotes, DataSourceType
from source.product.question_suggestions import build_data_aware_question_suggestions
from source.product.store_factory import create_investigation_store


st.set_page_config(page_title="Analytica · Datasets", layout="wide")

if "investigation_store" not in st.session_state:
    st.session_state.investigation_store = create_investigation_store()

store = st.session_state.investigation_store

st.title("Datasets")
st.caption("Upload CSV datasets, profile them, and use them in investigations.")

with st.sidebar:
    st.subheader("Upload dataset")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    name = st.text_input("Name")
    description = st.text_area("Description", height=80)
    tags = st.text_input("Tags", placeholder="metric, operations")
    sep = st.selectbox("Separator", [",", ";", "\t"], index=0)
    encoding = st.selectbox("Encoding", ["utf-8", "utf-8-sig", "cp1251"], index=0)
    if st.button("Upload CSV dataset", disabled=uploaded is None or not name.strip(), width="stretch"):
        raw = uploaded.getvalue()
        df = pd.read_csv(io.BytesIO(raw), sep=sep, encoding=encoding)
        source = DataSource(
            name=name.strip(),
            data_source_type=DataSourceType.CSV,
            description=description or None,
            tags=[item for item in tags.split(",")],
            metadata={"filename": uploaded.name},
        )
        saved = store.create_data_source(source)
        store.save_data_source_profile(saved.data_source_id, profile_dataframe(df))
        st.session_state.workspace_df = df
        st.success("Dataset uploaded")

sources = store.list_data_sources()
if not sources:
    st.info("No datasets yet.")
    st.stop()

table = pd.DataFrame(
    [
        {
            "name": source.name,
            "type": source.data_source_type.value,
            "status": source.status.value,
            "created_at": source.created_at.isoformat(),
            "linked_investigations": len(source.linked_investigation_ids),
            "tags": ", ".join(source.tags),
        }
        for source in sources
    ]
)
st.dataframe(table, width="stretch", hide_index=True)

for source in sources:
    with st.container(border=True):
        st.subheader(source.name)
        st.caption(f"{source.data_source_type.value} · {source.status.value} · `{source.data_source_id}`")
        if source.description:
            st.markdown(source.description)
        if source.tags:
            st.caption("Tags: " + ", ".join(source.tags))

        investigations = store.list_investigations()
        options = {"Select investigation": None}
        options.update({item.title: item.investigation_id for item in investigations})
        selected = st.selectbox("Link to investigation", list(options.keys()), key=f"link-{source.data_source_id}")
        if st.button("Link dataset", key=f"link-btn-{source.data_source_id}", disabled=options[selected] is None):
            store.link_data_source_to_investigation(options[selected], source.data_source_id)
            st.rerun()

        with st.expander("Profile", expanded=False):
            try:
                profile = store.get_data_source_profile(source.data_source_id)
                st.metric("Rows", profile.row_count)
                st.metric("Columns", profile.column_count)
                st.markdown("Columns")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "name": column.name,
                                "dtype": column.dtype,
                                "nullable": column.nullable,
                                "unique_count": column.unique_count,
                                "sample_values": ", ".join(map(str, column.sample_values)),
                            }
                            for column in profile.columns
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )
                st.markdown("Missing values")
                st.json(profile.missing_summary)
                if profile.numeric_summary:
                    st.markdown("Numeric summary")
                    st.json(profile.numeric_summary)
                if profile.categorical_summary:
                    st.markdown("Categorical summary")
                    st.json(profile.categorical_summary)
                if profile.sampled_rows:
                    st.markdown("Sample rows")
                    st.dataframe(pd.DataFrame(profile.sampled_rows), width="stretch", hide_index=True)
            except KeyError:
                st.info("No profile saved for this source.")

        with st.expander("Usage Context", expanded=False):
            context = build_data_source_usage_context(store, source.data_source_id)
            schema = context.schema_summary
            st.caption(
                f"{schema.get('row_count', 'unknown')} rows · "
                f"{schema.get('column_count', 'unknown')} columns · {context.status.value}"
            )
            if context.column_summaries:
                st.markdown("Inferred column roles")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "column": column.name,
                                "dtype": column.dtype,
                                "role": column.inferred_role.value,
                                "nullable": column.nullable,
                                "unique_count": column.unique_count,
                                "notes": "; ".join(column.notes),
                            }
                            for column in context.column_summaries
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )
            if context.caveats:
                st.markdown("Caveats")
                for caveat in context.caveats:
                    st.warning(caveat)
            if context.linked_investigation_ids:
                st.markdown("Linked investigations")
                for investigation_id in context.linked_investigation_ids:
                    st.caption(investigation_id)
            if context.previous_questions:
                st.markdown("Previous questions")
                for question in context.previous_questions:
                    st.markdown(f"- {question}")
            st.markdown("Suggested investigations")
            for suggestion in build_data_aware_question_suggestions(usage_context=context, limit=6):
                st.caption(suggestion)

        with st.expander("Semantic Notes", expanded=False):
            notes = store.get_data_source_semantic_notes(source.data_source_id)
            st.caption("Human-written notes that help the agent understand business meaning.")
            try:
                profile = store.get_data_source_profile(source.data_source_id)
                context = build_data_source_usage_context(store, source.data_source_id)
                inferred_roles = {column.name: column.inferred_role.value for column in context.column_summaries}
                profile_columns = [column.name for column in profile.columns]
            except KeyError:
                inferred_roles = {}
                profile_columns = [note.column_name for note in notes.column_notes]

            with st.form(f"semantic-source-{source.data_source_id}"):
                source_description = st.text_area(
                    "Source description",
                    value=notes.source_description or "",
                    height=80,
                    key=f"source-desc-{source.data_source_id}",
                )
                business_context = st.text_area(
                    "Business context",
                    value=notes.business_context or "",
                    height=100,
                    key=f"business-context-{source.data_source_id}",
                )
                global_caveats = st.text_area(
                    "Global caveats",
                    value="\n".join(notes.global_caveats),
                    height=90,
                    key=f"global-caveats-{source.data_source_id}",
                )
                if st.form_submit_button("Save source notes", width="stretch"):
                    saved_notes = DataSourceSemanticNotes(
                        data_source_id=source.data_source_id,
                        source_description=source_description or None,
                        business_context=business_context or None,
                        global_caveats=[item for item in global_caveats.splitlines()],
                        column_notes=notes.column_notes,
                    )
                    store.save_data_source_semantic_notes(saved_notes)
                    st.rerun()

            note_by_column = {note.column_name: note for note in notes.column_notes}
            for column_name in profile_columns:
                note = note_by_column.get(column_name, ColumnSemanticNote(column_name=column_name))
                with st.container(border=True):
                    st.markdown(f"**{column_name}** · inferred `{inferred_roles.get(column_name, 'unknown')}`")
                    with st.form(f"semantic-column-{source.data_source_id}-{column_name}"):
                        display_name = st.text_input("Display name", value=note.display_name or "")
                        description_text = st.text_area("Description", value=note.description or "", height=70)
                        business_meaning = st.text_area(
                            "Business meaning",
                            value=note.business_meaning or "",
                            height=70,
                        )
                        role_values = [role.value for role in ColumnSemanticRole]
                        semantic_role = st.selectbox(
                            "Semantic role",
                            role_values,
                            index=role_values.index(note.semantic_role.value),
                        )
                        caveats = st.text_area("Caveats", value="\n".join(note.caveats), height=70)
                        examples = st.text_area("Examples", value="\n".join(note.examples), height=70)
                        save_col, clear_col = st.columns(2)
                        with save_col:
                            save_column = st.form_submit_button("Save column note", width="stretch")
                        with clear_col:
                            clear_column = st.form_submit_button("Clear column note", width="stretch")
                        if save_column:
                            store.update_column_semantic_note(
                                source.data_source_id,
                                column_name,
                                ColumnSemanticNote(
                                    column_name=column_name,
                                    display_name=display_name or None,
                                    description=description_text or None,
                                    business_meaning=business_meaning or None,
                                    semantic_role=ColumnSemanticRole(semantic_role),
                                    caveats=[item for item in caveats.splitlines()],
                                    examples=[item for item in examples.splitlines()],
                                ),
                            )
                            st.rerun()
                        if clear_column:
                            store.delete_column_semantic_note(source.data_source_id, column_name)
                            st.rerun()
