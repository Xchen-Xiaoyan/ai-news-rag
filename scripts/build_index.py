"""Build BM25 and dense indexes. Usage: python scripts/build_index.py [embed_model ...]"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag import config
from src.rag.index import build_bm25, build_dense, passage_text


def main(models: list[str]):
    config.INDEX.mkdir(parents=True, exist_ok=True)
    docs = pd.read_parquet(config.DOCS, columns=["doc_id", "title"])
    chunks = pd.read_parquet(config.CHUNKS)
    texts = passage_text(chunks, docs)
    print(f"{len(texts):,} passages")

    t = time.time()
    build_bm25(texts, config.INDEX)
    print(f"bm25 built in {time.time() - t:.0f}s")

    for m in models:
        t = time.time()
        build_dense(texts, m, config.INDEX)
        print(f"{m} built in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1:] or [config.EMBED_MODEL])
