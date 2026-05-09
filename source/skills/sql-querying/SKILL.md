---
name: sql_querying
description: Read-only SQL-style querying over the current pandas DataFrame through an in-memory SQLite table.
allowed-tools: list_dataframe_tables, describe_dataframe_table, check_dataframe_sql, query_dataframe_sql
---

# SQL Querying

## When To Use
Use this skill when the user asks for SQL, table-style querying, filters, joins within the available table, grouped aggregations, or asks to "query" the CSV/DataFrame.

## Workflow
1. Call `list_dataframe_tables`.
2. Call `describe_dataframe_table` for the relevant table.
3. Write a read-only SQLite query using only returned table and column names.
4. Call `check_dataframe_sql`.
5. If valid, call `query_dataframe_sql`; if invalid, fix the query and check again.
6. Use SQL result output for the final answer.

## Rules
- Only SELECT, WITH, or PRAGMA queries are allowed.
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, REPLACE, TRUNCATE, ATTACH, DETACH, VACUUM, or REINDEX.
- Do not query all columns unless the user explicitly asks for raw sample rows.
- Respect configured row limits.
- Do not assume the table name; get it from `list_dataframe_tables`.
