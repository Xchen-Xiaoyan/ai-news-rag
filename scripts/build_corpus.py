"""Select the RAG subset, clean article bodies, deduplicate, and chunk.

Outputs data/processed/docs.parquet (one row per article) and chunks.parquet.
"""
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag import config
from src.rag.clean import extract_body
from src.rag.chunking import chunk_text

LABEL_COLS = ["industry_labels_strict", "technology_labels", "orgs_clean_v2", "pred_sentiment", "label"]


def title_key(title: str) -> str:
    # Strip outlet suffixes: "Headline | Site", "Headline - Site", "Headline – Site", "Headline — Site"
    t = re.split(r"\s[|\-–—]\s", title)[0]
    return re.sub(r"[^a-z0-9]", "", t.lower())


def main():
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    cols = ["url", "date", "title", "analysis_text", "final_score", "topic_id"]
    df = pd.read_parquet(config.CORPUS_RAW, columns=cols)
    hc = pd.read_parquet(config.HIGH_CONF_RAW, columns=["url"] + LABEL_COLS)
    hc = hc.drop_duplicates("url").rename(columns={
        "industry_labels_strict": "industries", "technology_labels": "technologies",
        "orgs_clean_v2": "orgs", "pred_sentiment": "sentiment", "label": "topic_label"})
    print(f"corpus: {len(df):,}  high_conf: {len(hc):,}")

    df = df[(df.final_score >= config.MIN_FINAL_SCORE) | df.url.isin(hc.url)].copy()
    print(f"selected (final_score >= {config.MIN_FINAL_SCORE} or high_conf): {len(df):,}")

    tqdm.pandas(desc="cleaning")
    df["body"] = df.progress_apply(lambda r: extract_body(r.analysis_text, r.title), axis=1)
    df = df.drop(columns="analysis_text")
    df["body_words"] = df.body.str.split().str.len()
    df = df[df.body_words >= 80]
    print(f"after cleaning (body >= 80 words): {len(df):,}")

    # Syndicated stories appear under many URLs; keep the labeled copy if any, else the longest.
    df["in_high_conf"] = df.url.isin(hc.url)
    df["tkey"] = df.title.map(title_key)
    df["bkey"] = df.body.str[:300].str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    df = df.sort_values(["in_high_conf", "body_words"], ascending=False)
    df = df.drop_duplicates("tkey").drop_duplicates("bkey").drop(columns=["tkey", "bkey"])
    print(f"after dedup: {len(df):,}")

    df = df.merge(hc, on="url", how="left")
    df["source"] = df.url.map(lambda u: urlparse(u).netloc.removeprefix("www."))
    df = df.sort_values("date").reset_index(drop=True)
    df.insert(0, "doc_id", [f"d{i:06d}" for i in range(len(df))])

    rows = []
    for r in tqdm(df.itertuples(), total=len(df), desc="chunking"):
        for k, text in enumerate(chunk_text(r.body)):
            rows.append((f"{r.doc_id}_c{k:02d}", r.doc_id, k, text))
    chunks = pd.DataFrame(rows, columns=["chunk_id", "doc_id", "chunk_idx", "text"])

    df.to_parquet(config.DOCS, index=False)
    chunks.to_parquet(config.CHUNKS, index=False)
    w = chunks.text.str.split().str.len()
    print(f"docs: {len(df):,}  chunks: {len(chunks):,}  words/chunk: median {w.median():.0f}, "
          f"p95 {w.quantile(.95):.0f}  high_conf docs: {df.in_high_conf.sum():,}")


if __name__ == "__main__":
    main()
