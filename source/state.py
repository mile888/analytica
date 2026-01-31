import pandas as pd
from typing import Any, List, Optional, Literal, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


class AgentState(TypedDict, total=False):
    query: str
    df: pd.DataFrame
    chat_history: List[BaseMessage]

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