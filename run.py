"""Quick CLI runner for the analytics agent pipeline.

Usage:
    python run.py
"""
import pandas as pd
from source.agent import run_once


import os
import sys
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    data_path = "data/train.csv"

    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at '{data_path}'. Please ensure the file exists.")
        sys.exit(1)

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    query = "Какие метрики и срезы нужны, чтобы понять падение продаж по категориям?"

    try:
        out = run_once(df, query)
    except Exception as e:
        print(f"Critical execution error: {e}")
        sys.exit(1)

    print("=== FINAL ANSWER ===")
    print(out["final_answer"])

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
