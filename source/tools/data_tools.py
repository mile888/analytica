from typing import Any

from source.func import detect_engine
from source.engine import create_engine


def inspect_schema_tool(df: Any, engine: str = "auto") -> dict:
    """
    Inspect dataset schema using selected compute engine.
    """
    eng_name = detect_engine(df, engine)
    eng = create_engine(eng_name)
    table = eng.ensure_table(df)
    schema = eng.schema_text(table)

    return {
        "engine": eng_name,
        "df": table,
        "schema": schema,
    }
