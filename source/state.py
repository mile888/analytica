from typing import Annotated, Any, List, Optional, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

EngineName = Literal["pandas", "polars", "spark"]
UseCase = Literal["data_analytics", "business_analytics"]


class AgentState(TypedDict, total=False):
    query: str
    df: Any
    messages: Annotated[list[BaseMessage], add_messages]

    use_case: UseCase
    needs_data: bool
    engine: EngineName
    schema: str

    plan: str
    code: str
    exec_error: Optional[str]
    result: Any
    result_preview: str

    result_kind: Literal["plot", "dataframe", "series", "scalar", "error"]
    result_facts: str

    critic_verdict: Literal["OK", "RETRY"]
    critic_feedback: str
    attempts: int
    max_attempts: int

    final_answer: str
