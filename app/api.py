"""REST API.  Run: uvicorn app.api:app --port 8000   (docs at /docs)"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.service import EXAMPLES, get_service

app = FastAPI(title="AI Industry News RAG", version="1.0",
              description="Cited answers about AI's impact on industries, from 33k news articles (2022-2026).")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    k: int = Field(8, ge=1, le=12)
    date_from: str | None = Field(None, description="YYYY-MM-DD")
    date_to: str | None = Field(None, description="YYYY-MM-DD")
    industries: list[str] = Field(default_factory=list)


@app.on_event("startup")
def warm_up():
    get_service()


@app.get("/health")
def health():
    s = get_service()
    return {"status": "ok", "docs": s.n_docs, "chunks": s.n_chunks, "llm": s.llm.name,
            "date_range": [str(d) for d in s.date_range], "llm_calls_left_today": s.budget.limit - s.budget.used}


@app.get("/examples")
def examples():
    return {"examples": list(EXAMPLES), "industries": get_service().industries}


@app.post("/ask")
def ask(req: AskRequest):
    try:
        return get_service().ask(req.question, k=req.k, date_from=req.date_from, date_to=req.date_to,
                                 industries=req.industries)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
