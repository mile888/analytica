from __future__ import annotations

from typing import Any

import pandas as pd


def deterministic_investigation_fallback(question: str, df: Any) -> dict[str, Any] | None:
    """Small product fallback for obvious tabular investigations.

    This does not replace the agent. It keeps the Investigation Workspace useful
    when the LLM returns prose without invoking analytics tools.
    """

    if not isinstance(df, pd.DataFrame) or df.empty:
        return None

    q = (question or "").lower()
    metric_col = _pick_column(
        df,
        q,
        [
            ("profit", ("profit", "приб")),
            ("sales", ("sales", "продаж", "выруч")),
            ("revenue", ("revenue", "выруч")),
        ],
    )
    dimension_col = _pick_column(
        df,
        q,
        [
            ("ship mode", ("ship mode", "ship", "достав", "режим доставки", "кораб")),
            ("segment", ("segment", "сегмент")),
            ("category", ("category", "категор")),
            ("region", ("region", "регион")),
            ("city", ("city", "город")),
            ("customer", ("customer", "клиент")),
        ],
    )

    if not metric_col:
        coverage = _metric_coverage_response(question, df)
        if coverage:
            return coverage

    if not metric_col or not dimension_col:
        return None
    if not pd.api.types.is_numeric_dtype(df[metric_col]):
        return None

    grouped = (
        df.groupby(dimension_col, dropna=False)[metric_col]
        .agg(["count", "sum", "mean", "median"])
        .sort_values("sum", ascending=False)
        .reset_index()
    )
    grouped["share_of_total"] = grouped["sum"] / grouped["sum"].sum() if grouped["sum"].sum() else 0.0

    top = grouped.iloc[0]
    bottom = grouped.iloc[-1]
    total = float(grouped["sum"].sum())

    result_preview = grouped.to_string(index=False)
    finding = (
        f"`{dimension_col}` заметно различается по `{metric_col}`: "
        f"максимальная сумма у `{top[dimension_col]}` ({top['sum']:.2f}), "
        f"минимальная сумма у `{bottom[dimension_col]}` ({bottom['sum']:.2f}). "
        f"Среднее значение: {top['mean']:.2f} против {bottom['mean']:.2f}."
    )
    summary = (
        f"Расчет сгруппировал `{metric_col}` по `{dimension_col}`. "
        f"Общая сумма `{metric_col}`: {total:.2f}. "
        "Таблица показывает количество строк, сумму, среднее, медиану и долю от общей суммы."
    )
    code = (
        f"result = (df.groupby({dimension_col!r}, dropna=False)[{metric_col!r}]\n"
        "    .agg(['count', 'sum', 'mean', 'median'])\n"
        "    .sort_values('sum', ascending=False)\n"
        "    .reset_index())\n"
        "result['share_of_total'] = result['sum'] / result['sum'].sum() if result['sum'].sum() else 0.0"
    )
    timeline = [
        {
            "tool": "deterministic_pandas_fallback",
            "status": "ok",
            "dimension": dimension_col,
            "metric": metric_col,
            "rows": int(len(df)),
        }
    ]

    return {
        "final_answer": summary,
        "code": code,
        "result_preview": result_preview,
        "result_base64": "",
        "exec_error": None,
        "engine": "pandas",
        "needs_data": True,
        "use_case": "data_analytics",
        "loaded_skills": ["data-analysis", "csv-dataframe-analysis", "reporting"],
        "tool_timeline": timeline,
        "sql_metadata": {},
        "structured_report": {
            "question": question,
            "summary": summary,
            "key_findings": [finding],
            "evidence": [f"Grouped `{metric_col}` by `{dimension_col}` over {len(df)} rows."],
            "limitations": ["Это описательная агрегация по текущему датасету; она не доказывает причинно-следственную связь."],
            "artifacts": [],
            "next_steps": [
                f"Сравнить `{metric_col}` по другим измерениям: category, region или city.",
                "Проверить отдельно суммы и средние значения: они могут вести к разным выводам.",
            ],
            "generated_code": code,
            "loaded_skills": ["data-analysis", "csv-dataframe-analysis", "reporting"],
            "tool_timeline": timeline,
            "sql_metadata": {},
        },
        "trace_metadata": {"fallback": "deterministic_pandas", "metric": metric_col, "dimension": dimension_col},
        "critic_verdict": "",
        "critic_feedback": "",
        "artifacts": [
            {
                "artifact_type": "table",
                "title": f"{metric_col} by {dimension_col}",
                "content": grouped.to_dict(orient="records"),
                "visibility": "user",
                "pinned": True,
                "metadata": {"preview": result_preview},
            }
        ],
    }


def _pick_column(
    df: pd.DataFrame,
    question: str,
    candidates: list[tuple[str, tuple[str, ...]]],
) -> str | None:
    columns = {str(col).lower(): str(col) for col in df.columns}
    for canonical, markers in candidates:
        if not any(marker in question for marker in markers):
            continue
        for lowered, original in columns.items():
            if lowered == canonical or canonical in lowered:
                return original
        for marker in markers:
            if not marker.isascii() or len(marker) < 4:
                continue
            for lowered, original in columns.items():
                if marker in lowered:
                    return original
    return None


def _metric_coverage_response(question: str, df: pd.DataFrame) -> dict[str, Any] | None:
    q = (question or "").lower()
    requested_metric = ""
    if "приб" in q or "profit" in q:
        requested_metric = "profit"
    elif "выруч" in q or "revenue" in q:
        requested_metric = "revenue"
    elif "продаж" in q or "sales" in q:
        requested_metric = "sales"
    if not requested_metric:
        return None

    numeric_columns = [str(col) for col in df.select_dtypes(include="number").columns]
    available_columns = [str(col) for col in df.columns]
    suggested = "Sales" if "Sales" in available_columns else (numeric_columns[0] if numeric_columns else "")
    finding = (
        f"Cannot answer the requested `{requested_metric}` question from the current dataset: "
        f"no matching `{requested_metric}` column was found."
    )
    summary = (
        f"The Investigation could not compute `{requested_metric}` because the dataset does not contain "
        f"a matching metric column. Available numeric columns: {', '.join(numeric_columns) or 'none'}."
    )
    next_steps = ["Upload a dataset that includes profit/margin if the decision requires profitability."]
    if suggested:
        next_steps.append(f"Run a related Investigation using `{suggested}` instead.")
    timeline = [
        {
            "tool": "data_coverage_check",
            "status": "metric_missing",
            "requested_metric": requested_metric,
            "available_numeric_columns": numeric_columns,
        }
    ]
    return {
        "final_answer": summary,
        "code": "",
        "result_preview": "",
        "result_base64": "",
        "exec_error": None,
        "engine": "pandas",
        "needs_data": True,
        "use_case": "data_coverage",
        "loaded_skills": ["data-analysis", "csv-dataframe-analysis", "reporting"],
        "tool_timeline": timeline,
        "sql_metadata": {},
        "structured_report": {
            "question": question,
            "summary": summary,
            "key_findings": [finding],
            "evidence": [f"Available columns: {', '.join(available_columns)}"],
            "limitations": [f"`{requested_metric}` is not present in the current dataset."],
            "artifacts": [],
            "next_steps": next_steps,
            "generated_code": "",
            "loaded_skills": ["data-analysis", "csv-dataframe-analysis", "reporting"],
            "tool_timeline": timeline,
            "sql_metadata": {},
        },
        "trace_metadata": {
            "fallback": "data_coverage_check",
            "requested_metric": requested_metric,
            "available_numeric_columns": numeric_columns,
        },
        "critic_verdict": "",
        "critic_feedback": "",
        "artifacts": [
            {
                "artifact_type": "validation",
                "title": "Data coverage check",
                "content": timeline,
                "visibility": "technical",
                }
        ],
    }
