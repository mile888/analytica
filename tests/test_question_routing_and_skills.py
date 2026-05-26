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

def test_context_builder_only_includes_current_investigation_messages() -> None:

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
