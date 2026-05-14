---
name: code-execution-safety
description: Safety policy for generated Python analysis code executed against the current dataset.
---

# Code Execution Safety

## When To Use
Use this skill before writing non-trivial Python analysis code or when fixing executor errors.

## Executor Contract
Generated code runs with:
- `df`: current dataset table.
- `pd`: pandas.
- `plt`: matplotlib pyplot when available.
- `to_pandas`: helper for engine conversion.

The final answer value must be assigned to `result`.

## Rules
- Do not read files, write files, export data, access network, start subprocesses, or import unsafe modules.
- Do not use `open`, `eval`, `exec`, `compile`, `input`, or `__import__`.
- Do not call `read_csv`, `read_excel`, `read_parquet`, `to_csv`, `to_excel`, `to_parquet`, or `to_sql`.
- Allowed imports are controlled by `ANALYTICA_ALLOWED_CODE_IMPORTS`.
- Keep outputs compact: scalar, small DataFrame/Series, or matplotlib Figure.
