"""Retrieval ablation: does the gold article come back, and how high? (No LLM calls.)

Writes eval/results/retrieval_summary.csv, retrieval_summary.md and per-question details.
"""
import gc
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag import config
from src.rag.evaluation import EVAL, doc_ranks, load_testset, retrieval_metrics
from src.rag.retriever import Retriever

SMALL, BASE = "BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5"

# (experiment, variant, embed model, mode, rerank)
CONFIGS = [
    ("method", "w200", SMALL, "bm25", False),
    ("method", "w200", SMALL, "dense", False),
    ("method", "w200", SMALL, "hybrid", False),
    ("method", "w200", SMALL, "bm25", True),
    ("method", "w200", SMALL, "dense", True),
    ("method", "w200", SMALL, "hybrid", True),
    ("chunk size", "w100", SMALL, "hybrid", True),
    ("chunk size", "w350", SMALL, "hybrid", True),
    ("embedder", "w200", BASE, "dense", False),
    ("embedder", "w200", BASE, "hybrid", True),
]


def available(variant, embed):
    _, index_dir = config.variant_paths(variant)
    return (index_dir / f"dense_{embed.split('/')[-1]}.npy").exists()


def main():
    items = [q for q in load_testset() if q["type"] != "unanswerable"]
    print(f"{len(items)} answerable questions")
    out_dir = EVAL / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, details = [], []
    groups = dict.fromkeys((v, e) for _, v, e, _, _ in CONFIGS)
    for variant, embed in groups:
        if not available(variant, embed):
            print(f"skip {variant}/{embed}: index not built")
            continue
        retriever = Retriever(embed_model=embed, variant=variant)
        for exp, v, e, mode, rerank in CONFIGS:
            if (v, e) != (variant, embed):
                continue
            name = f"{mode}{'+rerank' if rerank else ''} | {v} | {e.split('/')[-1]}"
            per_q, t0 = [], time.time()
            for q in tqdm(items, desc=name, leave=False):
                hits = retriever.search(q["question"], k=10, mode=mode, rerank=rerank)
                ranks = doc_ranks(hits, q["gold_doc_ids"])
                m = retrieval_metrics(ranks)
                per_q.append({**m, "type": q["type"]})
                details.append({"config": name, "id": q["id"], "type": q["type"], "ranks": ranks,
                                "top3": [h["title"][:80] for h in hits[:3]]})
            latency = (time.time() - t0) / len(items)
            df = pd.DataFrame(per_q)
            row = {"experiment": exp, "config": name, **df.drop(columns="type").mean().round(3).to_dict(),
                   "single hit@5": round(df[df.type == "single"]["hit@5"].mean(), 3),
                   "multi all@10": round(df[df.type == "multi"]["all@10"].mean(), 3),
                   "latency_s": round(latency, 3)}
            rows.append(row)
            print(f"{name:45s} hit@1 {row['hit@1']:.3f}  hit@5 {row['hit@5']:.3f}  "
                  f"mrr {row['mrr']:.3f}  multi all@10 {row['multi all@10']:.3f}  {latency:.2f}s/q")
        del retriever
        gc.collect()
        torch.cuda.empty_cache()

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "retrieval_summary.csv", index=False)
    cols = ["experiment", "config", "hit@1", "hit@5", "hit@10", "mrr", "multi all@10", "latency_s"]
    (out_dir / "retrieval_summary.md").write_text(
        f"Answerable questions: {len(items)}\n\n" + summary[cols].to_markdown(index=False), encoding="utf-8")
    with open(out_dir / "retrieval_details.jsonl", "w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
