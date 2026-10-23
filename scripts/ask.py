"""Command-line Q&A.

  python scripts/ask.py "How is AI changing radiology?"
  python scripts/ask.py "..." --retrieve-only --mode bm25 --no-rerank
  python scripts/ask.py "..." --industry healthcare_medical --from 2024-01-01
"""
import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag.retriever import Filters, Retriever


def main():
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("-k", type=int, default=8)
    p.add_argument("--mode", default="hybrid", choices=["bm25", "dense", "hybrid"])
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--retrieve-only", action="store_true")
    p.add_argument("--provider", default=None, help="gemini | ollama (default: $LLM_PROVIDER)")
    p.add_argument("--from", dest="date_from")
    p.add_argument("--to", dest="date_to")
    p.add_argument("--industry", action="append", default=[])
    a = p.parse_args()
    load_dotenv()

    t = time.time()
    retriever = Retriever()
    print(f"[loaded in {time.time() - t:.1f}s]\n")
    filters = Filters(date_from=a.date_from, date_to=a.date_to, industries=a.industry)
    kw = dict(k=a.k, mode=a.mode, rerank=not a.no_rerank, filters=filters)

    t = time.time()
    if a.retrieve_only:
        hits = retriever.search(a.question, **kw)
        print(f"[retrieved in {time.time() - t:.2f}s]")
        for i, h in enumerate(hits, 1):
            print(f"\n[{i}] {h['score']:.3f} {h['title'][:100]}\n    {h['source']} {h['date']}\n    {h['text'][:300]}...")
        return

    from src.rag.llm import get_llm
    from src.rag.pipeline import RAGPipeline
    out = RAGPipeline(retriever, get_llm(a.provider)).answer(a.question, **kw)
    print(f"[answered in {time.time() - t:.1f}s by {out.get('llm')}]\n\n{out['answer']}\n\nSources:")
    for i, h in enumerate(out["sources"], 1):
        mark = "*" if i in out["cited"] else " "
        print(f" {mark}[{i}] {h['title'][:90]} ({h['source']}, {h['date']})\n      {h['url']}")


if __name__ == "__main__":
    main()
