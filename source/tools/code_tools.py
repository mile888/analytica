from typing import Any

from source.func import safe_exec, preview_result_and_facts


def execute_code_tool(code: str, df: Any, engine: str) -> dict:
    """
    Execute analyst-authored code against the current dataset and return result metadata.

    This is the project-level execution tool used by the Deep Agent. It keeps data in
    memory, exposes controlled analysis libraries, and requires the final value in
    `result` so the app can preview tables, scalars, and matplotlib figures.
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
