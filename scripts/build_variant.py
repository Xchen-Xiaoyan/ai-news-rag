"""Build extra indexes for the ablations (reuses the cleaned bodies in docs.parquet).

  python scripts/build_variant.py --chunks w100          # different chunk size, default embedder
  python scripts/build_variant.py --embed BAAI/bge-base-en-v1.5   # different embedder, default chunks
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag import config
from src.rag.chunking import chunk_text
from src.rag.index import build_bm25, build_dense, passage_text


def main(variant: str, embed: str):
    chunks_path, index_dir = config.variant_paths(variant)
    index_dir.mkdir(parents=True, exist_ok=True)
    docs = pd.read_parquet(config.DOCS, columns=["doc_id", "title", "body"])

    if not chunks_path.exists():
        target, max_words = config.CHUNK_VARIANTS[variant]
        rows = []
        for r in tqdm(docs.itertuples(), total=len(docs), desc=f"chunking {variant}"):
            for k, text in enumerate(chunk_text(r.body, target=target, max_words=max_words)):
                rows.append((f"{r.doc_id}_c{k:02d}", r.doc_id, k, text))
        pd.DataFrame(rows, columns=["chunk_id", "doc_id", "chunk_idx", "text"]).to_parquet(chunks_path, index=False)

    chunks = pd.read_parquet(chunks_path)
    texts = passage_text(chunks, docs)
    print(f"{variant}: {len(texts):,} passages")
    if not (index_dir / "bm25").exists():
        build_bm25(texts, index_dir)
    t = time.time()
    build_dense(texts, embed, index_dir)
    print(f"{embed} on {variant} built in {time.time() - t:.0f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", default=config.DEFAULT_VARIANT, choices=list(config.CHUNK_VARIANTS))
    p.add_argument("--embed", default=config.EMBED_MODEL)
    a = p.parse_args()
    main(a.chunks, a.embed)
