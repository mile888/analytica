from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from fastapi.responses import RedirectResponse
import pandas as pd

from source.agent import run_once
from source.dataframe import read_csv_dataset

app = FastAPI(title="Analytica Data Agent")

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

class AnalyzeRequest(BaseModel):
    query: str = Field(..., min_length=1)
    data: Optional[List[Dict[str, Any]]] = None
    csv_path: Optional[str] = None
    csv_sep: str = ","
    csv_encoding: str = "utf-8"
    engine: str = "auto"
    thread_id: Optional[str] = None


class AnalyzeResponse(BaseModel):
    final_answer: str
    result_preview: str
    code: str
    critic_verdict: str
    critic_feedback: str
    exec_error: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    try:
        if req.data is not None:
            df = pd.DataFrame(req.data)
        elif req.csv_path:
            df = read_csv_dataset(req.csv_path, sep=req.csv_sep, encoding=req.csv_encoding)
        else:
            raise ValueError("Provide either `data` records or `csv_path`.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid data: {e}")

    try:
        result = run_once(df=df, query=req.query, engine=req.engine, thread_id=req.thread_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    return AnalyzeResponse(**result)
