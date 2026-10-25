"""Write the minimal data bundle the demo needs to data/deploy/ (upload it as a PRIVATE HF dataset).

Keeps only what retrieval reads: slim article metadata (no full bodies), chunks, BM25 and the
bge-small vectors, plus the demo answer cache so example questions answer instantly.

  python scripts/export_deploy_data.py
  huggingface-cli upload <user>/ai-news-rag-data data/deploy --repo-type dataset --private
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag import config

OUT = config.DATA / "deploy"


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "processed").mkdir(parents=True)
    (OUT / "index").mkdir()
    (OUT / "cache").mkdir()

    cols = ["doc_id", "url", "date", "title", "source", "in_high_conf", "industries", "sentiment"]
    pd.read_parquet(config.DOCS, columns=cols).to_parquet(OUT / "processed" / "docs.parquet", index=False)
    shutil.copy(config.CHUNKS, OUT / "processed" / "chunks.parquet")
    shutil.copytree(config.INDEX / "bm25", OUT / "index" / "bm25")
    dense = f"dense_{config.EMBED_MODEL.split('/')[-1]}.npy"
    shutil.copy(config.INDEX / dense, OUT / "index" / dense)
    cache = config.DATA / "cache" / "demo_cache.sqlite"
    if cache.exists():
        shutil.copy(cache, OUT / "cache" / cache.name)

    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    for f in sorted(OUT.rglob("*")):
        if f.is_file():
            print(f"{f.stat().st_size / 1e6:8.1f} MB  {f.relative_to(OUT)}")
    print(f"{total / 1e6:8.1f} MB  total")


if __name__ == "__main__":
    main()
