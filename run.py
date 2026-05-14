"""Quick CLI runner for the DeepAgents analytics agent.

Usage:
    python run.py --csv path/to/file.csv --query "Опиши структуру данных"
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from source.agent import run_once
from source.config import DEFAULT_CLI_QUERY, DEFAULT_DATA_PATH
from source.dataframe import read_csv_dataset


from dotenv import load_dotenv
load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Analytica data agent from CLI.")
    parser.add_argument(
        "--csv",
        default=str(DEFAULT_DATA_PATH),
        help="CSV path. Relative paths are resolved from the project root.",
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_CLI_QUERY,
        help="Analytical question for the agent.",
    )
    parser.add_argument(
        "--engine",
        default="auto",
        choices=["auto", "pandas", "polars", "spark"],
        help="Compute engine used by analysis tools.",
    )
    parser.add_argument("--sep", default=",", help="CSV separator.")
    parser.add_argument("--encoding", default="utf-8", help="CSV encoding.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        df: pd.DataFrame = read_csv_dataset(args.csv, sep=args.sep, encoding=args.encoding)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)

    try:
        out = run_once(df, args.query, engine=args.engine)
    except Exception as e:
        print(f"Critical execution error: {e}")
        sys.exit(1)

    print("=== FINAL ANSWER ===")
    print(out["final_answer"])

    print("\n=== SELECTED SKILLS ===")
    print(out.get("selected_skills", []))

    print("\n=== SELECTED TOOLS ===")
    print(out.get("selected_tools", []))

    print("\n=== GENERATED CODE ===")
    print(out["code"])

    print("\n=== EXEC ERROR ===")
    print(out["exec_error"])

    print("\n=== RESULT PREVIEW ===")
    print(out["result_preview"])

    # Plot output (if any) is already encoded as a base64 data-URI
    # and can be rendered directly in a browser or Streamlit UI.
    b64 = out.get("result_base64", "")
    if b64:
        print("\n=== PLOT (base64 data-URI, first 120 chars) ===")
        print(b64[:120] + "...")
    else:
        print("\n=== No plot generated ===")
