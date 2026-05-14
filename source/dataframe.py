from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd

from source.config import MAX_SQL_ROWS, SQL_TABLE_NAME, resolve_project_path


_READ_ONLY_SQL_RE = re.compile(r"^\s*(select|with|pragma)\b", re.IGNORECASE | re.DOTALL)
_BLOCKED_SQL_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|detach|vacuum|reindex)\b",
    re.IGNORECASE,
)


class DataFrameSqlResult(NamedTuple):
    data: pd.DataFrame
    row_count: int
    truncated: bool
    table_name: str


def read_csv_dataset(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV from an absolute path or a path relative to the project root."""
    csv_path = resolve_project_path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not csv_path.is_file():
        raise ValueError(f"CSV path is not a file: {csv_path}")
    read_kwargs = dict(kwargs)
    try:
        return pd.read_csv(csv_path, **read_kwargs)
    except pd.errors.ParserError:
        retry_kwargs = dict(read_kwargs)
        retry_kwargs.setdefault("escapechar", chr(92))
        try:
            return pd.read_csv(csv_path, **retry_kwargs)
        except pd.errors.ParserError:
            retry_kwargs.setdefault("on_bad_lines", "skip")
            return pd.read_csv(csv_path, **retry_kwargs)


def dataframe_profile(df: pd.DataFrame, *, sample_rows: int = 5) -> dict[str, Any]:
    """Return compact schema/profile metadata for prompts, UI, and notebooks."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("dataframe_profile expects a pandas DataFrame")
    return {
        "rows": int(len(df)),
        "columns_count": int(len(df.columns)),
        "columns": [str(col) for col in df.columns],
        "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
        "missing": {str(col): int(value) for col, value in df.isna().sum().items()},
        "sample": df.head(sample_rows).to_dict(orient="records"),
    }


def _safe_table_name(table_name: str | None = None) -> str:
    configured_name = SQL_TABLE_NAME or "data"
    name = (table_name or configured_name).strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        return configured_name if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", configured_name) else "data"
    return name


def dataframe_to_sqlite(df: pd.DataFrame, table_name: str | None = None) -> tuple[sqlite3.Connection, str]:
    """Load a DataFrame into an in-memory SQLite database for read-only querying."""
    table = _safe_table_name(table_name)
    conn = sqlite3.connect(":memory:")
    df.to_sql(table, conn, index=False, if_exists="replace")
    return conn, table


def list_sql_tables(df: pd.DataFrame, table_name: str | None = None) -> list[str]:
    _conn, table = dataframe_to_sqlite(df, table_name)
    try:
        return [table]
    finally:
        _conn.close()


def sql_table_schema(df: pd.DataFrame, table_name: str | None = None, sample_rows: int = 3) -> str:
    conn, table = dataframe_to_sqlite(df, table_name)
    try:
        schema_rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
        sample = pd.read_sql_query(f'SELECT * FROM "{table}" LIMIT {int(sample_rows)}', conn)
    finally:
        conn.close()

    lines = [f'Table "{table}" columns:']
    for _, name, dtype, notnull, default, pk in schema_rows:
        flags = []
        if notnull:
            flags.append("not null")
        if pk:
            flags.append("primary key")
        suffix = f" ({', '.join(flags)})" if flags else ""
        lines.append(f"- {name}: {dtype or 'TEXT'}{suffix}")
    lines.append("")
    lines.append(f"Sample rows ({len(sample)}):")
    lines.append(sample.to_string(index=False))
    return "\n".join(lines)


def validate_read_only_sql(query: str) -> str:
    """Lightweight SQL query checker inspired by LangChain SQL-agent workflow."""
    sql = (query or "").strip().rstrip(";")
    if not sql:
        raise ValueError("SQL query is empty.")
    if not _READ_ONLY_SQL_RE.match(sql):
        raise ValueError("Only read-only SELECT/WITH/PRAGMA queries are allowed.")
    if _BLOCKED_SQL_RE.search(sql):
        raise ValueError("DML/DDL statements are not allowed for DataFrame SQL queries.")
    if ";" in sql:
        raise ValueError("Only a single SQL statement is allowed.")
    return sql


def run_dataframe_sql(
    df: pd.DataFrame,
    query: str,
    table_name: str | None = None,
    max_rows: int = MAX_SQL_ROWS,
) -> pd.DataFrame:
    """Execute a checked read-only SQLite query against the DataFrame."""
    sql = validate_read_only_sql(query)
    conn, _table = dataframe_to_sqlite(df, table_name)
    try:
        out = pd.read_sql_query(sql, conn)
    finally:
        conn.close()
    return out.head(max_rows)


def run_dataframe_sql_with_metadata(
    df: pd.DataFrame,
    query: str,
    table_name: str | None = None,
    max_rows: int = MAX_SQL_ROWS,
) -> DataFrameSqlResult:
    """Execute read-only SQL and return bounded results plus metadata."""
    sql = validate_read_only_sql(query)
    conn, table = dataframe_to_sqlite(df, table_name)
    try:
        out = pd.read_sql_query(sql, conn)
    finally:
        conn.close()
    row_count = int(len(out))
    return DataFrameSqlResult(
        data=out.head(max_rows),
        row_count=row_count,
        truncated=row_count > max_rows,
        table_name=table,
    )
