from pathlib import Path

import pandas as pd

from source import agent as agent_module
from source.config import PROJECT_ROOT
from source.product.branch_workspace import BranchWorkspace, branch_dtos_for_investigation, upsert_branch_for_intent
from source.product.investigation import InvestigationMessage, InvestigationMessageRole
from source.product.llm_reasoning import apply_llm_reasoning_layer, sanitize_user_visible_text
from source.product.non_analytical import sanitize_non_analytical_text
from source.product.question_routing import BranchAction, QuestionIntentType, classify_question_intent, decide_routing, is_non_analytical_intent
from source.product.run_service import InvestigationRunService, _build_conversation_context
from source.product.store import InvestigationStore


REQUIRED_SKILLS = {
    "branch-continuation-policy",
    "multi-dataset-reasoning",
    "executive-analyst-reasoning",
    "analyst-response-style",
}


def test_required_deep_agent_skills_exist_and_are_exposed() -> None:
    for skill in REQUIRED_SKILLS:
        assert (PROJECT_ROOT / "source" / "skills" / skill / "SKILL.md").is_file()

    sources = {Path(item).parts[-1] for item in agent_module._deep_agent_skill_sources()}
    assert REQUIRED_SKILLS <= sources


def test_system_prompt_mentions_high_value_skills_briefly() -> None:
    prompt = agent_module._build_system_prompt(["inspect_dataset_schema"])

    for skill in REQUIRED_SKILLS:
        assert skill in prompt
        assert prompt.count(skill) == 1
    assert len(prompt.splitlines()) < 15


def test_multi_dataset_intents_route_global_before_active_branch() -> None:
    examples = {
        "Can these datasets be joined reliably?": QuestionIntentType.MULTI_DATASET_JOINABILITY,
        "What fields could potentially connect these datasets?": QuestionIntentType.MULTI_DATASET_JOINABILITY,
        "What important business questions cannot currently be answered because of missing links between these datasets?": QuestionIntentType.MULTI_DATASET_MISSING_LINKS,
        "If you had to design a unified analytics warehouse from these datasets, what shared entities would you introduce?": QuestionIntentType.MULTI_DATASET_WAREHOUSE_DESIGN,
        "What are the 3 most important cross-dataset insights?": QuestionIntentType.MULTI_DATASET_SEMANTIC_REASONING,
    }

    for question, expected in examples.items():
        decision = decide_routing(question, has_active_context=True)
        assert decision.question_intent_type == expected
        assert decision.branch_action == BranchAction.GLOBAL
        assert decision.artifact_context_action == "ignore_active_branch"


def test_followups_continue_but_strategic_questions_do_not() -> None:
    assert decide_routing("compare against SF", has_active_context=True).branch_action == BranchAction.CONTINUE
    assert decide_routing("explain this chart", has_active_context=True).branch_action == BranchAction.CONTINUE

    strategic = decide_routing("If you were a product manager, what would you investigate first and why?", has_active_context=True)
    assert strategic.question_intent_type == QuestionIntentType.STRATEGIC_RECOMMENDATION
    # Strategic questions now create independent analytical branches (not global bypass)
    # because they should be answered with computed evidence from the dataset.
    assert strategic.branch_action == BranchAction.CREATE


def test_non_analytical_intents_escape_active_branch() -> None:
    examples = {
        "найди сходство с фильмом лалалэнд": QuestionIntentType.CREATIVE_ANALOGY,
        "find similarities with La La Land": QuestionIntentType.CREATIVE_ANALOGY,
        "what should I test next?": QuestionIntentType.META_PROJECT_QUESTION,
        "why did the previous answer look wrong?": QuestionIntentType.META_PROJECT_QUESTION,
    }

    for question, expected in examples.items():
        decision = decide_routing(question, has_active_context=True)
        assert decision.question_intent_type == expected
        assert is_non_analytical_intent(decision.question_intent_type)
        assert decision.branch_action == BranchAction.NONE
        assert decision.execution_mode == "conversational"
        assert decision.artifact_context_action == "ignore_active_branch"


def test_ambiguous_creative_comparison_asks_clarification_not_grouped_fallback() -> None:
    decision = decide_routing("compare this with La La Land", has_active_context=True)

    assert decision.question_intent_type == QuestionIntentType.SEMANTIC_ANALOGY
    assert decision.branch_action == BranchAction.NONE
    assert decision.execution_mode == "conversational"


def test_global_routing_drops_stale_active_branch_context() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Top cities by Sales")
    store.update_investigation_metadata(
        investigation.investigation_id,
        {
            "conversation_state": {
                "active_branch_id": "grouped::sales::city",
                "active_metric": "Sales",
                "active_dimension": "City",
                "active_transformation_result": {"metric": "Sales", "dimension": "City"},
                "distribution_state": {"metric": "Sales"},
            }
        },
    )

    context = _build_conversation_context(
        store,
        investigation.investigation_id,
        active_question="Can these datasets be joined reliably?",
    )
    state = context["conversation_state"]

    assert state["branch_action"] == "global"
    assert "active_branch_id" not in state
    assert "active_metric" not in state
    assert "active_transformation_result" not in state
    assert "distribution_state" not in state


def test_new_histogram_request_resets_stale_grouped_context() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Z_Revenue by Education")
    store.update_investigation_metadata(
        investigation.investigation_id,
        {
            "conversation_state": {
                "active_branch_id": "grouped::z_revenue::education",
                "active_metric": "Z_Revenue",
                "active_dimension": "Education",
                "distribution_state": {"metric": "Z_Revenue", "chart_type": "histogram"},
            }
        },
    )

    context = _build_conversation_context(
        store,
        investigation.investigation_id,
        active_question="Build histogram of Sales in LA",
    )
    state = context["conversation_state"]

    assert state["branch_action"] == "create"
    assert state["context_policy"] == "reset_analytical"
    assert "active_metric" not in state
    assert "active_dimension" not in state
    assert "distribution_state" not in state


def test_non_analytical_context_drops_stale_active_branch_context() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Income by Education")
    store.update_investigation_metadata(
        investigation.investigation_id,
        {
            "conversation_state": {
                "active_branch_id": "grouped::income::education",
                "active_metric": "Income",
                "active_dimension": "Education",
            }
        },
    )

    context = _build_conversation_context(
        store,
        investigation.investigation_id,
        active_question="найди сходство с фильмом лалалэнд",
    )
    state = context["conversation_state"]

    assert state["branch_action"] == "none"
    assert state["question_intent_type"] == "creative_analogy"
    assert "active_branch_id" not in state
    assert "active_metric" not in state
    assert "active_dimension" not in state
    assert state["context_policy"] == "conversational"


def test_branch_titles_for_business_and_multi_dataset_intents_are_meaningful() -> None:
    workspace = BranchWorkspace()
    workspace = upsert_branch_for_intent(workspace, intent_type="business_risk", created_from_query="What hidden risks do you see?")
    workspace = upsert_branch_for_intent(workspace, intent_type="multi_dataset_joinability", created_from_query="Can these datasets be joined?")
    workspace = upsert_branch_for_intent(workspace, intent_type="multi_dataset_warehouse_design", created_from_query="Design a warehouse")

    store = InvestigationStore()
    investigation = store.create_investigation("multi")
    investigation.metadata["branch_workspace"] = workspace.to_payload()

    titles = {item["title"]: item for item in branch_dtos_for_investigation(investigation)}
    assert "Business risks" in titles
    assert "Dataset joinability" in titles
    assert "Warehouse design" in titles
    assert titles["Dataset joinability"]["intent_type"] == "multi_dataset_joinability"


def test_sanitizer_blocks_active_branch_hijack_phrases() -> None:
    text = sanitize_user_visible_text(
        "The active comparison is still `Income` by `Marital_Status`.\n"
        "Judge it by the leading groups.\n"
        "The answer is not reliably joinable yet."
    )

    assert "active comparison" not in text.lower()
    assert "judge it by the leading" not in text.lower()
    assert "not reliably joinable" in text.lower()


def test_high_level_reasoning_avoids_weak_metric_names(monkeypatch) -> None:
    def fail_llm(_name):
        raise ValueError("no provider")

    monkeypatch.setattr("source.product.llm_reasoning.make_llm", fail_llm)
    result = apply_llm_reasoning_layer(
        question="What hidden business risks do you see?",
        output={
            "summary": "Computed grouped result.",
            "artifacts": [
                {
                    "type": "table",
                    "metadata": {"metric": "Row ID"},
                    "content": [{"Row ID": 1, "total": 10}, {"Row ID": 2, "total": 5}],
                }
            ],
        },
    )

    text = result["summary"]
    assert "Row ID" not in text
    assert "`sum`" not in text
    # Post-cleanup: local synthesis restates evidence or uses neutral labels
    assert "Computed grouped result" not in text or "Validate" in text


def test_creative_query_after_active_branch_does_not_use_dataframe_summary() -> None:
    df = pd.DataFrame(
        {
            "Income": [50_000, 75_000, 80_000, 45_000],
            "Education": ["Bachelor", "Master", "Master", "High school"],
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("income by education")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    message = store.add_investigation_message(
        InvestigationMessage(
            investigation_id=investigation.investigation_id,
            content="найди сходство с фильмом лалалэнд",
        )
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=message.message_id)

    assistant = [
        item.content
        for item in store.list_investigation_messages(investigation.investigation_id)
        if item.role == InvestigationMessageRole.ASSISTANT
    ][-1]
    lowered = assistant.lower()
    assert "metaphor" in lowered or "creative interpretation" in lowered
    assert "income" not in lowered
    assert "education" not in lowered
    assert "using average" not in lowered


def test_histogram_explicit_metric_overrides_stale_metric() -> None:
    df = pd.DataFrame(
        {
            "Z_Revenue": [100, 200, 300, 400],
            "Sales": [10, 15, 20, 25],
            "Education": ["Bachelor", "Master", "Bachelor", "Master"],
            "City": ["LA", "LA", "SF", "LA"],
        }
    )
    store = InvestigationStore()
    investigation = store.create_investigation("Z_Revenue by Education")
    store.update_investigation_metadata(
        investigation.investigation_id,
        {
            "conversation_state": {
                "active_metric": "Z_Revenue",
                "active_dimension": "Education",
                "distribution_state": {"metric": "Z_Revenue", "chart_type": "histogram"},
            }
        },
    )
    message = store.add_investigation_message(
        InvestigationMessage(
            investigation_id=investigation.investigation_id,
            content="Build histogram of Sales in LA",
        )
    )

    InvestigationRunService(store).run_investigation(investigation.investigation_id, df=df, message_id=message.message_id)

    updated = store.get_investigation(investigation.investigation_id)
    assistant = [
        item.content
        for item in store.list_investigation_messages(investigation.investigation_id)
        if item.role == InvestigationMessageRole.ASSISTANT
    ][-1]
    lowered = assistant.lower()
    assert "sales" in lowered
    assert "z_revenue" not in lowered
    assert "education" not in lowered
    assert any(
        (artifact.metadata.get("metric") == "Sales" or (artifact.metadata.get("metadata") or {}).get("metric") == "Sales")
        for artifact in updated.artifacts
    )
    assert " by `" not in lowered


def test_meta_query_after_active_branch_stays_project_level() -> None:
    df = pd.DataFrame({"Sales": [10, 20, 30], "City": ["A", "A", "B"]})
    store = InvestigationStore()
    investigation = store.create_investigation("top cities by sales")
    service = InvestigationRunService(store)

    service.run_investigation(investigation.investigation_id, df=df)
    message = store.add_investigation_message(
        InvestigationMessage(
            investigation_id=investigation.investigation_id,
            content="what should I test next?",
        )
    )
    service.run_investigation(investigation.investigation_id, df=df, message_id=message.message_id)

    assistant = [
        item.content
        for item in store.list_investigation_messages(investigation.investigation_id)
        if item.role == InvestigationMessageRole.ASSISTANT
    ][-1]
    lowered = assistant.lower()
    assert "test" in lowered
    assert "agent" in lowered or "demo" in lowered
    assert "sales by city" not in lowered
    assert "using average" not in lowered


def test_joinability_global_query_ignores_stale_grouped_metric() -> None:
    store = InvestigationStore()
    investigation = store.create_investigation("Z_Revenue by Education")
    store.update_investigation_metadata(
        investigation.investigation_id,
        {
            "conversation_state": {
                "active_metric": "Z_Revenue",
                "active_dimension": "Education",
                "active_grouped_payload": {"metric": "Z_Revenue", "dimension": "Education"},
                "distribution_state": {"metric": "Z_Revenue"},
            }
        },
    )

    context = _build_conversation_context(
        store,
        investigation.investigation_id,
        active_question="Can these datasets be joined reliably?",
    )
    state = context["conversation_state"]

    assert state["context_policy"] == "reset_global"
    assert state["branch_action"] == "global"
    assert "active_metric" not in state
    assert "active_dimension" not in state
    assert "active_grouped_payload" not in state
    assert "distribution_state" not in state


def test_non_analytical_sanitizer_blocks_grouped_text() -> None:
    text = sanitize_non_analytical_text(
        "`Income` by `Education` is led by `Master` using average `Income`.",
        "find similarities with La La Land",
        QuestionIntentType.CREATIVE_ANALOGY,
    )

    assert "is led by" not in text
    assert "using average" not in text
    assert "metaphor" in text.lower()


# =========================================================================
# Regression: cross-investigation context isolation
# =========================================================================

def test_movie_dataset_question_is_not_classified_as_creative_analogy() -> None:
    """Questions about movie/film data must be analytical, not creative."""
    analytical_questions = [
        "Compare movie and TV show distributions",
        "How many movies by country?",
        "Top genres by movie count",
        "Show movie rating distribution",
        "Which country has the most films?",
        "Film count by genre",
    ]
    for question in analytical_questions:
        intent = classify_question_intent(question)
        assert intent != QuestionIntentType.CREATIVE_ANALOGY, (
            f"'{question}' was misclassified as CREATIVE_ANALOGY"
        )
        assert intent != QuestionIntentType.SEMANTIC_ANALOGY, (
            f"'{question}' was misclassified as SEMANTIC_ANALOGY"
        )


def test_genuine_creative_requests_still_classified_correctly() -> None:
    """Named creative entity references must still route to creative analogy."""
    creative_questions = [
        "find similarities with La La Land",
        "найди сходство с фильмом лалалэнд",
        "use a metaphor to describe the data",
        "what analogy best describes this pattern?",
    ]
    for question in creative_questions:
        intent = classify_question_intent(question)
        assert intent in {QuestionIntentType.CREATIVE_ANALOGY, QuestionIntentType.SEMANTIC_ANALOGY}, (
            f"'{question}' was NOT classified as creative: got {intent}"
        )


def test_cross_investigation_isolation_no_la_la_land_leakage() -> None:
    """Investigation B must not be influenced by investigation A's context."""
    df_a = pd.DataFrame({
        "Sales": [100, 200, 300],
        "Category": ["Tech", "Furniture", "Supplies"],
    })
    df_b = pd.DataFrame({
        "show_id": ["s1", "s2", "s3", "s4", "s5", "s6"],
        "type": ["Movie", "TV Show", "Movie", "TV Show", "Movie", "Movie"],
        "title": ["Title A", "Title B", "Title C", "Title D", "Title E", "Title F"],
        "country": ["US", "India", "UK", "US", "Japan", "India"],
        "listed_in": ["Dramas", "Comedies", "Action", "Dramas", "Thrillers", "Comedies"],
        "release_year": [2020, 2019, 2021, 2020, 2022, 2021],
    })

    store = InvestigationStore()

    investigation_a = store.create_investigation("find similarities with La La Land")
    service_a = InvestigationRunService(store)
    service_a.run_investigation(investigation_a.investigation_id, df=df_a)

    investigation_b = store.create_investigation("Compare movie and TV show distributions")
    service_b = InvestigationRunService(store)
    service_b.run_investigation(investigation_b.investigation_id, df=df_b)

    state_b = store.get_investigation(investigation_b.investigation_id).metadata.get("conversation_state", {})
    messages_b = [
        item.content
        for item in store.list_investigation_messages(investigation_b.investigation_id)
        if item.role == InvestigationMessageRole.ASSISTANT
    ]

    for msg in messages_b:
        lowered = msg.lower()
        assert "la la land" not in lowered, f"Investigation B mentions La La Land: {msg[:200]}"
        assert "metaphor" not in lowered, f"Investigation B uses metaphor language: {msg[:200]}"
        assert "creative interpretation" not in lowered, f"Investigation B uses creative language: {msg[:200]}"

    assert state_b.get("active_metric") != "Sales"


def test_new_investigation_has_isolated_artifacts_and_state() -> None:
    """Artifacts and state from investigation A must not appear in investigation B."""
    store = InvestigationStore()

    inv_a = store.create_investigation("Top cities by sales")
    store.update_investigation_metadata(
        inv_a.investigation_id,
        {
            "conversation_state": {
                "active_metric": "Sales",
                "active_dimension": "City",
                "active_branch_id": "grouped::sales::city",
            }
        },
    )

    inv_b = store.create_investigation("Movie analysis")

    context_b = _build_conversation_context(
        store,
        inv_b.investigation_id,
        active_question="Compare movie and TV show distributions",
    )
    state_b = context_b["conversation_state"]

    assert state_b.get("active_metric") is None or state_b.get("active_metric") == ""
    assert state_b.get("active_dimension") is None or state_b.get("active_dimension") == ""
    assert state_b.get("active_branch_id") is None or state_b.get("active_branch_id") == ""

    artifacts_b = store.get_investigation(inv_b.investigation_id).artifacts
    assert len(artifacts_b) == 0

    messages_b = store.list_investigation_messages(inv_b.investigation_id)
    assert len(messages_b) == 0


def test_context_builder_only_includes_current_investigation_messages() -> None:
    """Context builder must only include messages from the current investigation."""
    store = InvestigationStore()

    inv_a = store.create_investigation("Investigation A question")
    store.add_investigation_message(
        InvestigationMessage(
            investigation_id=inv_a.investigation_id,
            content="La La Land comparison result",
            role=InvestigationMessageRole.ASSISTANT,
        )
    )

    inv_b = store.create_investigation("Investigation B question")
    store.add_investigation_message(
        InvestigationMessage(
            investigation_id=inv_b.investigation_id,
            content="Movie distribution analysis",
            role=InvestigationMessageRole.ASSISTANT,
        )
    )

    context_b = _build_conversation_context(
        store,
        inv_b.investigation_id,
        active_question="Next analytical question",
    )

    messages_text = " ".join(
        str(msg.get("content", "")) for msg in context_b.get("messages", [])
    )
    assert "La La Land" not in messages_text
    assert "Investigation A" not in messages_text
