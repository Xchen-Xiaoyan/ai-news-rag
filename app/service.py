"""Shared Q&A service for the API and the Streamlit UI.

Adds what a public demo needs on top of RAGPipeline: lazy data download, a demo-only answer
cache, a global daily cap on LLM calls (free-tier quota), and retrieval-only fallback when the
LLM is unavailable.
"""
import os
import sys
import threading
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag import config  # noqa: E402

# question -> short label for the UI
EXAMPLES = {
    "How have US export controls affected Nvidia's AI chip sales to China?": "Nvidia & export controls",
    "Why did Klarna start rehiring human customer service agents after its AI push?": "Klarna rehiring humans",
    "How is generative AI changing radiology and medical imaging?": "AI in radiology",
    "What did the EEOC allege in its first lawsuit over AI in hiring?": "First AI-hiring lawsuit",
    "AI 对客服和呼叫中心岗位有什么影响?": "AI 对客服岗位的影响",
    "What was Apple's Vision Pro revenue in Q3 2027?": "Vision Pro 2027 (not in archive)",  # shows abstention
}

SNIPPET_CHARS = 320  # the UI shows short excerpts + links, never full articles
CPU_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def ensure_data():
    """On a fresh deployment, pull the index from a private HF dataset (HF_DATA_REPO + HF_TOKEN)."""
    if config.CHUNKS.exists():
        return
    repo = os.getenv("HF_DATA_REPO")
    if not repo:
        raise RuntimeError(f"{config.CHUNKS} missing and HF_DATA_REPO not set")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(config.DATA),
                      token=os.getenv("HF_TOKEN"))


class DailyBudget:
    def __init__(self, limit: int):
        self.limit, self.day, self.used = limit, date.today(), 0
        self.lock = threading.Lock()

    def take(self) -> bool:
        with self.lock:
            if date.today() != self.day:
                self.day, self.used = date.today(), 0
            if self.used >= self.limit:
                return False
            self.used += 1
            return True


class QAService:
    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        ensure_data()

        import pandas as pd
        from src.rag.llm import CachedLLM, get_llm
        from src.rag.pipeline import RAGPipeline
        from src.rag.retriever import Retriever

        device = os.getenv("RAG_DEVICE")  # None -> cuda if available
        t = time.time()
        # CPU serving (REPORT.md §6): bge-reranker-base takes ~11 s for 20 pairs on 2 vCPUs, MiniLM-L6
        # ~1.5 s at -3.4 pp Hit@1. 20 candidates score the same as 50 for both.
        cpu = device == "cpu"
        rerank_model = os.getenv("RAG_RERANK_MODEL", CPU_RERANKER if cpu else config.RERANK_MODEL)
        self.n_candidates = int(os.getenv("RAG_RERANK_CANDIDATES", "20"))
        if cpu:
            import torch
            torch.set_num_threads(int(os.getenv("RAG_THREADS", os.cpu_count() or 2)))
        self.retriever = Retriever(device=device, rerank_model=rerank_model)
        self.reranker_name = rerank_model.split("/")[-1]
        llm = get_llm(os.getenv("LLM_PROVIDER", "gemini"), cache=False)
        self.llm = CachedLLM(llm, config.DATA / "cache" / "demo_cache.sqlite")
        self.rag = RAGPipeline(self.retriever, self.llm)
        self.budget = DailyBudget(int(os.getenv("MAX_DAILY_LLM_CALLS", "200")))
        self.load_seconds = round(time.time() - t, 1)

        docs = self.retriever.docs
        self.n_docs, self.n_chunks = len(docs), len(self.retriever.chunks)
        self.date_range = (docs.date.min().date(), docs.date.max().date())
        self.industries = sorted(pd.Series([i for xs in docs.industries.dropna() for i in xs]).unique())

    def _is_cached(self, question: str, hits: list[dict]) -> bool:
        return self.llm.contains(self.rag.system, self.rag.prompt_for(question, hits))

    def ask(self, question: str, k: int = 8, date_from: str | None = None, date_to: str | None = None,
            industries: list[str] | None = None) -> dict:
        from src.rag.retriever import Filters
        question = question.strip()[:500]
        filters = Filters(date_from=date_from, date_to=date_to, industries=industries or [])
        t0 = time.time()
        error = None
        try:
            queries, hits = self.rag.retrieve(question, k=k, filters=filters, n_candidates=self.n_candidates)
        except Exception as e:  # noqa: BLE001 - CJK rewrite needs the LLM; fall back to the raw question
            queries, hits = [question], self.retriever.search(question, k=k, filters=filters,
                                                              n_candidates=self.n_candidates)
            error = f"query rewrite failed: {e}"[:200]
        t_retrieve = time.time() - t0

        answer, cited, abstained = None, [], False
        cached = bool(hits) and self._is_cached(question, hits)
        if hits and (cached or self.budget.take()):
            try:
                out = self.rag.generate(question, hits)
                answer, cited, abstained = out["answer"], out["cited"], out["abstained"]
            except Exception as e:  # noqa: BLE001 - show retrieval results even if the LLM is down
                error = f"LLM unavailable: {e}"[:200]
        elif hits:
            error = "Daily LLM budget for this demo is used up; showing retrieved sources only."
        elif not hits:
            answer, abstained = "No matching articles for these filters.", True

        sources = [{**{k_: h[k_] for k_ in ["chunk_id", "title", "url", "source", "date", "score"]},
                    "snippet": h["text"][:SNIPPET_CHARS] + ("…" if len(h["text"]) > SNIPPET_CHARS else "")}
                   for h in hits]
        return {
            "question": question, "queries": queries, "answer": answer, "cited": cited,
            "abstained": abstained, "sources": sources, "cached": cached, "error": error,
            "llm": self.llm.name, "timing": {"retrieve_s": round(t_retrieve, 2),
                                             "total_s": round(time.time() - t0, 2)},
        }


_service: QAService | None = None
_lock = threading.Lock()


def get_service() -> QAService:
    global _service
    with _lock:
        if _service is None:
            _service = QAService()
        return _service
