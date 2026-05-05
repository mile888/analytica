from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from fastapi.responses import RedirectResponse
import pandas as pd

from source.agent import run_once

app = FastAPI(title="Analytica Data Agent")

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

class AnalyzeRequest(BaseModel):
    query: str = Field(..., min_length=1)
    data: List[Dict[str, Any]]


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
        df = pd.DataFrame(req.data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid data: {e}")

    try:
        result = run_once(df=df, query=req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    return AnalyzeResponse(**result)
