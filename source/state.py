from typing import Annotated, Any, List, Optional, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from source.llm.llm_config import EngineName
UseCase = Literal["data_analytics", "business_analytics"]


class AgentState(TypedDict, total=False):
    """LangGraph state for the analytics agent pipeline.

    Fields are grouped by lifecycle stage:
      • Inputs   — set by the caller (run_once).
      • Router   — populated by the router node.
      • Pipeline — filled during plan → code → exec → critic loop.
      • Output   — the final answer returned to the user.
    """

    # ── Inputs (set by caller) ───────────────────────────────────────
    query: str                                          # user's natural-language request
    df: Any                                             # source table (pandas / polars / spark)
    messages: Annotated[list[BaseMessage], add_messages] # unified conversation history
    engine: EngineName                                  # compute engine override ("auto" resolved earlier)

    # ── Router outputs ───────────────────────────────────────────────
    needs_data: bool                                    # True → data analytics path
    use_case: UseCase                                   # which pipeline branch to follow
    schema: str                                         # textual schema of the dataset

    # ── Pipeline internals ───────────────────────────────────────────
    plan: str                                           # step-by-step analysis plan
    code: str                                           # generated Python code
    exec_error: Optional[str]                           # execution error message (None if ok)
    result: Any                                         # raw execution result
    result_preview: str                                 # human-readable result preview
    result_kind: Literal["plot", "dataframe", "series", "scalar", "error"]
    result_facts: str                                   # structured facts about the result
    result_base64: str                                  # base64 data-URI for plot images

    # ── Critic loop ──────────────────────────────────────────────────
    critic_verdict: Literal["OK", "RETRY"]
    critic_feedback: str                                # feedback for codegen on retry
    attempts: int                                       # current retry count
    max_attempts: int                                   # retry budget

    # ── Final output ─────────────────────────────────────────────────
    final_answer: str                                   # markdown answer for the user
