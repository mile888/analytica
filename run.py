import pandas as pd
from source.agent import run_once
import matplotlib.pyplot as plt


if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")
    query = '/bar x=Category y=Sales agg=sum top=5 title=Top-5_categories_by_sales'
    out = run_once(df, query)

    print("=== FINAL ANSWER ===")
    print(out["final_answer"])

    print("\n=== GENERATED CODE ===")
    print(out["code"])

    print("\n=== EXEC ERROR ===")
    print(out["exec_error"])
    print("\n=== RESULT PREVIEW ===")
    print(out["result_preview"])
    plt.show()

