from __future__ import annotations

from typing import Any

import pandas as pd

from source.product.fallbacks.narration import _output, _timeline
from source.product.fallbacks.semantic_resolution import _normalize
from source.product.fallbacks.validation import _is_duplicate_question
from source.product.language_policy import ResponseLanguagePolicy


def _data_quality_response(question: str, df: pd.DataFrame) -> dict[str, Any]:
    if _is_duplicate_question(question):
        return _duplicate_orders_response(question, df)
    language = ResponseLanguagePolicy.from_message(question, protected_terms=df.columns)
    missing = df.isna().sum().sort_values(ascending=False)
    missing_rows = [
        {"column": str(col), "missing_values": int(count), "missing_rate": float(count / max(len(df), 1))}
        for col, count in missing.items()
        if int(count) > 0
    ]
    duplicate_count = int(df.duplicated().sum())
    numeric_columns = [str(col) for col in df.select_dtypes(include="number").columns]
    mentioned_columns = _mentioned_quality_columns(question, df)
    quality_findings = []
    if missing_rows:
        top_missing = missing_rows[0]
        quality_findings.append(
            f"У `{top_missing['column']}` больше всего missing values "
            f"({top_missing['missing_values']} rows, {top_missing['missing_rate']:.1%})."
            if language.is_russian
            else f"`{top_missing['column']}` has the most missing values "
            f"({top_missing['missing_values']} rows, {top_missing['missing_rate']:.1%})."
        )
    else:
        quality_findings.append("Missing values в доступных строках не найдены." if language.is_russian else "No missing values were detected in the available rows.")
    if duplicate_count:
        quality_findings.append(
            f"Найдено {duplicate_count} duplicate row{'' if duplicate_count == 1 else 's'}."
            if language.is_russian
            else f"{duplicate_count} duplicate row{'' if duplicate_count == 1 else 's'} were detected."
        )
    else:
        quality_findings.append("Duplicate rows не найдены." if language.is_russian else "No duplicate rows were detected.")
    for column in mentioned_columns[:3]:
        missing_count = int(df[column].isna().sum())
        missing_rate = float(missing_count / max(len(df), 1))
        if language.is_russian:
            quality_findings.append(f"В `{column}` missing values: {missing_count} rows ({missing_rate:.1%}).")
        else:
            quality_findings.append(f"`{column}` has {missing_count} missing rows ({missing_rate:.1%}).")

    summary = (
        f"В датасете {len(df):,} строк и {len(df.columns):,} колонок. {quality_findings[0]} {quality_findings[1]}"
        if language.is_russian
        else f"The dataset has {len(df):,} rows and {len(df.columns):,} columns. {quality_findings[0]} {quality_findings[1]}"
    )
    if numeric_columns:
        summary += (
            f" Для reliability review важны numeric fields: {', '.join(numeric_columns[:4])}. "
            "Quality risks важны потому, что missingness может смещать segment comparisons, duplicates завышают counts/totals, "
            "а extreme numeric values искажают averages."
            if language.is_russian
            else f" The highest-value numeric fields for reliability review are {', '.join(numeric_columns[:4])}. "
            "Quality issues matter because missingness can bias segment comparisons, duplicates can inflate counts, "
            "and extreme numeric values can distort averages."
        )

    if mentioned_columns:
        target_bits = "; ".join(f"`{column}`={int(df[column].isna().sum())} missing" for column in mentioned_columns[:3])
        summary += f" Отдельно проверено: {target_bits}." if language.is_russian else f" Specifically checked: {target_bits}."
    treatment_note = _safe_missing_treatment_note(df, mentioned_columns, language)
    summary += " " + treatment_note

    result_rows = missing_rows[:20] or [{"column": str(col), "missing_values": 0, "missing_rate": 0.0} for col in df.columns[:10]]
    result_table = pd.DataFrame(result_rows)
    timeline = _timeline("data_quality_check", rows=len(df), columns=len(df.columns), duplicates=duplicate_count)
    return _output(
        question=question,
        summary=summary,
        findings=quality_findings,
        evidence=[f"Проверены missing values и duplicates на {len(df):,} строках."] if language.is_russian else [f"Checked missing values and duplicates across {len(df):,} rows."],
        limitations=[
            "Эта quality check описательная; domain-specific validity rules могут быть нужны отдельно.",
            "Колонки с высокой missingness ослабляют выводы, которые зависят от этих полей.",
        ] if language.is_russian else [
            "This quality check is descriptive; domain-specific validity rules may still be needed.",
            "Columns with heavy missingness can weaken conclusions that depend on those fields.",
        ],
        next_steps=[
            "Для numeric columns использовать median imputation и добавить missingness indicator; для categorical columns использовать отдельную категорию `Missing`.",
            "Проверить high-missing columns перед использованием в выводах.",
            "Решить, являются ли duplicate rows ожидаемыми records или data quality issue.",
            "Проверить outliers в важных numeric fields перед reliance on averages.",
        ] if language.is_russian else [
            "Use median imputation plus a missingness indicator for numeric columns; use an explicit `Missing` category for categorical columns.",
            "Review high-missing columns before using them in conclusions.",
            "Decide whether duplicate rows are expected records or data quality issues.",
            "Flag outliers in important numeric fields before relying on averages.",
        ],
        code="missing = df.isna().sum(); duplicate_count = df.duplicated().sum()",
        result_preview=result_table.to_string(index=False),
        timeline=timeline,
        artifacts=[
            {
                "artifact_type": "table",
                "title": "Dataset quality summary",
                "content": result_rows,
                "visibility": "user",
                "pinned": True,
                "metadata": {"duplicate_count": duplicate_count},
            }
        ],
        trace_metadata={"fallback": "data_quality_check", "analysis_type": "data_quality", "active_branch_type": "data_quality"},
    )


def _mentioned_quality_columns(question: str, df: pd.DataFrame) -> list[str]:
    text = _normalize(question)
    result: list[str] = []
    for column in df.columns:
        name = str(column)
        normalized = _normalize(name)
        if normalized and normalized in text:
            result.append(name)
    return result


def _safe_missing_treatment_note(df: pd.DataFrame, mentioned_columns: list[str], language: ResponseLanguagePolicy) -> str:
    targets = mentioned_columns or [str(col) for col in df.columns if int(df[col].isna().sum()) > 0][:3]
    if not targets:
        return "Обработка пропусков сейчас не требуется." if language.is_russian else "No missing-value treatment is needed right now."
    numeric = [column for column in targets if column in df.columns and pd.api.types.is_numeric_dtype(df[column])]
    categorical = [column for column in targets if column in df.columns and not pd.api.types.is_numeric_dtype(df[column])]
    if language.is_russian:
        parts = []
        if numeric:
            parts.append("для numeric fields безопаснее median imputation, потому что median устойчивее к outliers, чем mean")
        if categorical:
            parts.append("для categorical fields безопаснее отдельная категория `Missing`, потому что она не выдумывает несуществующее значение")
        return "Безопасная обработка: " + "; ".join(parts) + ". Не удаляйте строки автоматически, пока не ясно, что missingness случайная и малая."
    parts = []
    if numeric:
        parts.append("median imputation is safer for numeric fields because the median is more robust to outliers than the mean")
    if categorical:
        parts.append("an explicit `Missing` category is safer for categorical fields because it does not invent a real category")
    return "Safe treatment: " + "; ".join(parts) + ". Avoid dropping rows automatically until missingness is known to be small and random."


def _order_identifier_columns(df: pd.DataFrame) -> list[str]:
    result = []
    for col in df.columns:
        normalized = _normalize(str(col))
        if "order" in normalized and ("id" in normalized or "number" in normalized or normalized == "order"):
            result.append(str(col))
        elif any(marker in normalized for marker in ("order id", "order_id", "transaction id", "transaction_id", "заказ")):
            result.append(str(col))
    return result


def _duplicate_orders_response(question: str, df: pd.DataFrame) -> dict[str, Any]:
    language = ResponseLanguagePolicy.from_message(question, protected_terms=df.columns)
    exact_duplicate_rows = int(df.duplicated().sum())
    duplicate_families = int(df[df.duplicated(keep=False)].drop_duplicates().shape[0]) if exact_duplicate_rows else 0
    order_columns = _order_identifier_columns(df)
    rows: list[dict[str, Any]] = [
        {"check": "exact_duplicate_rows", "affected_rows": exact_duplicate_rows, "note": "Exact repeated rows are the strongest duplicate evidence."}
    ]
    order_note = "Order-like identifier column was not found, so I checked exact row duplicates only."
    if order_columns:
        order_col = order_columns[0]
        duplicated_order_rows = int(df[order_col].duplicated(keep=False).sum())
        duplicated_order_ids = int(df.loc[df[order_col].duplicated(keep=False), order_col].nunique(dropna=True))
        rows.append(
            {
                "check": f"repeated_{order_col}",
                "affected_rows": duplicated_order_rows,
                "duplicate_ids": duplicated_order_ids,
                "note": "Repeated order IDs can be normal multi-line orders.",
            }
        )
        if language.is_russian:
            order_note = (
                f"`{order_col}` выглядит как order-like identifier: {duplicated_order_ids:,} repeated ID value"
                f"{'' if duplicated_order_ids == 1 else 's'} на {duplicated_order_rows:,} строках. "
                "Это может быть нормальная структура line items, а не duplicate orders."
            )
        else:
            order_note = (
                f"`{order_col}` is the order-like identifier. It has {duplicated_order_ids:,} repeated ID value"
                f"{'' if duplicated_order_ids == 1 else 's'} across {duplicated_order_rows:,} rows. "
                "That may represent multi-line orders rather than duplicate orders."
            )
    if exact_duplicate_rows:
        duplicate_note = (
            f"Найдено {exact_duplicate_rows:,} exact duplicate row{'' if exact_duplicate_rows == 1 else 's'} "
            f"({duplicate_families:,} repeated row pattern{'' if duplicate_families == 1 else 's'})."
            if language.is_russian
            else f"{exact_duplicate_rows:,} exact duplicate row{'' if exact_duplicate_rows == 1 else 's'} were detected "
            f"({duplicate_families:,} repeated row pattern{'' if duplicate_families == 1 else 's'})."
        )
    else:
        duplicate_note = "Exact duplicate rows не найдены." if language.is_russian else "No exact duplicate rows were detected."
    summary = (
        f"Проверка duplicate orders: {duplicate_note} {order_note} "
        "Для влияния на метрики безопаснее удалять exact duplicate rows, а не все repeated order IDs, потому что repeated IDs часто означают отдельные line items."
        if language.is_russian
        else f"Duplicate-order check: {duplicate_note} {order_note} "
        "For metric impact, exact duplicate rows are safer to remove than every repeated order ID, because repeated IDs often mean separate line items."
    )
    evidence = [f"Проверил full-row duplicates и order-like ID columns на {len(df):,} строках."] if language.is_russian else [f"Checked full-row duplicates and order-like ID columns across {len(df):,} rows."]
    limitations = (
        ["Repeated order IDs не являются автоматической ошибкой: order-line datasets обычно содержат несколько строк на заказ.", "Near-duplicates требуют domain keys: customer, product, date и amount."]
        if language.is_russian
        else ["Repeated order IDs are not automatically bad data; order-line datasets normally contain several rows per order.", "Near-duplicates require domain keys such as customer, product, date, and amount."]
    )
    next_steps = (
        ["Оценить totals с exact duplicate rows и без них.", "Проверить repeated order IDs на identical line-item details перед удалением."]
        if language.is_russian
        else ["Estimate metric totals with and without exact duplicate rows.", "Inspect repeated order IDs for identical line-item details before deleting them."]
    )
    return _output(
        question=question,
        summary=summary,
        findings=[duplicate_note, order_note],
        evidence=evidence,
        limitations=limitations,
        next_steps=next_steps,
        code="exact_duplicate_rows = df.duplicated().sum(); repeated_order_ids = df[order_id].duplicated(keep=False).sum()",
        result_preview=pd.DataFrame(rows).to_string(index=False),
        timeline=_timeline("duplicate_order_check", rows=len(df), duplicates=exact_duplicate_rows),
        artifacts=[
            {
                "artifact_type": "table",
                "title": "Duplicate order checks",
                "content": rows,
                "visibility": "user",
                "pinned": True,
                "metadata": {"duplicate_rows": exact_duplicate_rows, "order_identifier_columns": order_columns},
            }
        ],
        trace_metadata={
            "fallback": "duplicate_order_check",
            "analysis_type": "data_quality",
            "active_branch_type": "data_quality",
            "active_quality_issue": "duplicates",
            "duplicate_rows": exact_duplicate_rows,
        },
    )


def _duplicate_impact_response(question: str, df: pd.DataFrame, metric_col: str | None, dimension_col: str | None) -> dict[str, Any]:
    language = ResponseLanguagePolicy.from_message(question, protected_terms=[item for item in (metric_col, dimension_col, *df.columns) if item])
    duplicate_mask = df.duplicated(keep=False)
    duplicate_excess_mask = df.duplicated(keep="first")
    duplicate_rows = int(duplicate_mask.sum())
    exact_duplicate_rows = int(duplicate_excess_mask.sum())
    duplicate_groups = []
    if dimension_col and duplicate_rows:
        duplicate_groups = (
            df.loc[duplicate_mask]
            .groupby(dimension_col, dropna=False)
            .size()
            .sort_values(ascending=False)
            .head(8)
            .reset_index(name="duplicate_rows")
            .to_dict(orient="records")
        )
    metric_note = ""
    result_rows = duplicate_groups or [{"duplicate_rows": duplicate_rows, "duplicate_rate": duplicate_rows / max(len(df), 1)}]
    if metric_col and metric_col in df.columns:
        original_total = float(pd.to_numeric(df[metric_col], errors="coerce").sum())
        dedup_total = float(pd.to_numeric(df.drop_duplicates()[metric_col], errors="coerce").sum())
        duplicate_sales = original_total - dedup_total
        metric_note = (
            f" Для `{metric_col}` total: {original_total:.2f} до exact-row deduplication и {dedup_total:.2f} после; "
            f"estimated inflation от exact duplicates: {duplicate_sales:.2f}."
            if language.is_russian
            else f" For `{metric_col}`, total is {original_total:.2f} before exact-row deduplication and {dedup_total:.2f} after; "
            f"the exact-duplicate inflation estimate is {duplicate_sales:.2f}."
        )
        result_rows = [
            {
                "metric": metric_col,
                "total_before_dedup": original_total,
                "total_after_exact_row_dedup": dedup_total,
                "estimated_duplicate_inflation": duplicate_sales,
                "exact_duplicate_excess_rows": exact_duplicate_rows,
            }
        ]
    if duplicate_rows:
        if dimension_col and duplicate_groups:
            top = duplicate_groups[0]
            summary = (
                f"Duplicates затрагивают {duplicate_rows:,} строк ({duplicate_rows / max(len(df), 1):.1%}). "
                f"Самая большая концентрация duplicates: `{dimension_col}` = `{top[dimension_col]}` ({top['duplicate_rows']} duplicate rows), "
                "поэтому group counts могут быть inflated, если эти записи не ожидаемы."
                if language.is_russian
                else f"Duplicates affect {duplicate_rows:,} rows ({duplicate_rows / max(len(df), 1):.1%}). "
                f"The largest duplicate concentration is in `{dimension_col}` = `{top[dimension_col]}` with {top['duplicate_rows']} duplicate rows, "
                "so group counts can be inflated if duplicates are not expected records."
            )
        else:
            summary = (
                f"Duplicates затрагивают {duplicate_rows:,} строк ({duplicate_rows / max(len(df), 1):.1%}). "
                "Они могут завышать counts и totals; averages обычно менее чувствительны, если duplicate records не сконцентрированы в отдельных группах."
                if language.is_russian
                else f"Duplicates affect {duplicate_rows:,} rows ({duplicate_rows / max(len(df), 1):.1%}). "
                "They can inflate record counts and totals; averages are less affected unless duplicate records are concentrated in specific groups."
            )
    else:
        summary = "Duplicate rows не найдены, поэтому duplicates не выглядят источником inflation в доступных данных." if language.is_russian else "No duplicate rows were detected, so duplicates do not appear to inflate important group counts in the available data."
    summary += metric_note
    order_columns = _order_identifier_columns(df)
    if order_columns:
        summary += (
            f" Я не трактовал каждый repeated `{order_columns[0]}` как inflation, потому что repeated order IDs могут быть нормальными line items."
            if language.is_russian
            else f" I did not treat every repeated `{order_columns[0]}` as metric inflation because repeated order IDs may be normal line items."
        )
    evidence = [f"Проверил exact duplicate rows на {len(df):,} records."] if language.is_russian else [f"Checked exact duplicate rows across {len(df):,} records."]
    limitations = ["Это проверка exact duplicate rows; near-duplicates или business-key duplicates могут оставаться."] if language.is_russian else ["This checks exact duplicate rows; near-duplicates or business-key duplicates may still exist."]
    next_steps = (
        [
            f"Сравнить group counts до и после deduplication{f' по `{dimension_col}`' if dimension_col else ''}.",
            f"Проверить, меняют ли duplicates totals для `{metric_col}`." if metric_col else "Проверить, меняют ли duplicates totals для основной метрики.",
        ]
        if language.is_russian
        else [
            f"Compare group counts before and after deduplication{f' by `{dimension_col}`' if dimension_col else ''}.",
            f"Check whether duplicates change totals for `{metric_col}`." if metric_col else "Check whether duplicates change totals for the primary numeric column.",
        ]
    )
    return _output(
        question=question,
        summary=summary,
        findings=[summary],
        evidence=evidence,
        limitations=limitations,
        next_steps=next_steps,
        code="deduped = df.drop_duplicates(); total_delta = df[metric].sum() - deduped[metric].sum()",
        result_preview=pd.DataFrame(result_rows).to_string(index=False),
        timeline=_timeline("duplicate_impact_check", rows=len(df), duplicates=duplicate_rows),
        artifacts=[
            {
                "artifact_type": "table",
                "title": "Duplicate impact summary",
                "content": result_rows,
                "visibility": "user",
                "pinned": True,
                "metadata": {"duplicate_rows": duplicate_rows, "metric": metric_col, "dimension": dimension_col},
            }
        ],
        trace_metadata={
            "fallback": "duplicate_impact_check",
            "analysis_type": "data_quality",
            "active_branch_type": "data_quality",
            "active_quality_issue": "duplicates",
            "duplicate_rows": duplicate_rows,
        },
    )


def _quality_issue_impact_response(question: str, df: pd.DataFrame, metric_col: str | None, dimension_col: str | None) -> dict[str, Any]:
    missing = df.isna().sum().sort_values(ascending=False)
    duplicate_count = int(df.duplicated().sum())
    metric_missing = int(df[metric_col].isna().sum()) if metric_col and metric_col in df.columns else 0
    dimension_missing = int(df[dimension_col].isna().sum()) if dimension_col and dimension_col in df.columns else 0
    top_missing_column = str(missing.index[0]) if len(missing) and int(missing.iloc[0]) else ""
    if duplicate_count and duplicate_count >= max(metric_missing, dimension_missing, int(missing.iloc[0]) if len(missing) else 0):
        summary = (
            f"Duplicates are the strongest quality risk: {duplicate_count:,} duplicate rows can inflate counts and totals, "
            "especially if the strongest conclusion depends on group volume."
        )
    elif metric_col and metric_missing:
        summary = (
            f"Missing values in `{metric_col}` are the strongest risk for the current conclusion: {metric_missing:,} rows lack `{metric_col}`, "
            "which can bias averages, rankings, and outlier checks."
        )
    elif dimension_col and dimension_missing:
        summary = (
            f"Missing values in `{dimension_col}` are the strongest risk: {dimension_missing:,} rows lack the grouping field, "
            "which can bias comparisons between groups."
        )
    elif top_missing_column:
        summary = (
            f"The main quality issue is missingness in `{top_missing_column}` ({int(missing.iloc[0]):,} rows). "
            "It matters most if this field is used for segmentation, filtering, or reporting."
        )
    else:
        summary = "No major missing-value or duplicate issue was detected in the available rows, so quality risk looks limited for the current conclusion."
    rows = [
        {"issue": "duplicate_rows", "affected_rows": duplicate_count},
        {"issue": f"missing_{metric_col}" if metric_col else "missing_metric", "affected_rows": metric_missing},
        {"issue": f"missing_{dimension_col}" if dimension_col else "missing_dimension", "affected_rows": dimension_missing},
    ]
    return _output(
        question=question,
        summary=summary,
        findings=[summary],
        evidence=[f"Checked duplicates and missingness across {len(df):,} records."],
        limitations=["This quality assessment is descriptive; domain-specific validation rules may still be required."],
        next_steps=["Re-run the strongest comparison after handling the highest-impact quality issue."],
        code="df.isna().sum(); df.duplicated().sum()",
        result_preview=pd.DataFrame(rows).to_string(index=False),
        timeline=_timeline("quality_impact_check", rows=len(df), duplicates=duplicate_count),
        artifacts=[
            {
                "artifact_type": "table",
                "title": "Quality issue impact",
                "content": rows,
                "visibility": "user",
                "pinned": True,
                "metadata": {"metric": metric_col, "dimension": dimension_col},
            }
        ],
        trace_metadata={"fallback": "quality_impact_check", "analysis_type": "data_quality"},
    )
