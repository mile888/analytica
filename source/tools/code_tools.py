from typing import Any

from source.func import safe_exec, preview_result_and_facts


def execute_code_tool(code: str, df: Any, engine: str) -> dict:
    """
    Execute generated code safely and return result metadata.
    """
    result, err = safe_exec(code, df, engine)
    kind, preview, facts, b64 = preview_result_and_facts(result, err)

    return {
        "result": result,
        "exec_error": err,
        "result_kind": kind,
        "result_preview": preview,
        "result_facts": facts,
        "result_base64": b64,
    }
