"""Test-set loading and metrics shared by the eval scripts."""
import json
from pathlib import Path

import pandas as pd

from . import config

EVAL = config.ROOT / "eval"


def load_testset(path: Path | None = None) -> list[dict]:
    """Load the final test set if it exists, else the draft with review decisions applied.

    In testset_review.csv, keep = n drops a question; edited question / reference_answer text wins.
    """
    final = EVAL / "testset.jsonl"
    path = path or (final if final.exists() else EVAL / "testset_draft.jsonl")
    items = [json.loads(line) for line in open(path, encoding="utf-8")]
    review = EVAL / "testset_review.csv"
    if path.name == "testset_draft.jsonl" and review.exists():
        rv = pd.read_csv(review, encoding="utf-8-sig", dtype=str).fillna("").set_index("id")
        out = []
        for it in items:
            if it["id"] in rv.index:
                r = rv.loc[it["id"]]
                if r.keep.strip().lower() in ("n", "no", "0"):
                    continue
                it["question"] = r.question.strip() or it["question"]
                it["reference_answer"] = r.reference_answer.strip() or it["reference_answer"]
            out.append(it)
        items = out
    # Hand-written questions (same JSON fields) are appended if present.
    extra = EVAL / "testset_manual.jsonl"
    if extra.exists():
        items += [json.loads(line) for line in open(extra, encoding="utf-8") if line.strip()]
    return items


def doc_ranks(hits: list[dict], gold: list[str]) -> list[int | None]:
    """1-based rank of each gold article in the (deduplicated) article order of the hits."""
    order = list(dict.fromkeys(h["doc_id"] for h in hits))
    return [order.index(g) + 1 if g in order else None for g in gold]


def retrieval_metrics(ranks: list[int | None], ks=(1, 3, 5, 10)) -> dict:
    found = [r for r in ranks if r is not None]
    first = min(found) if found else None
    m = {f"hit@{k}": float(first is not None and first <= k) for k in ks}
    m["mrr"] = 1.0 / first if first else 0.0
    m["all@10"] = float(len(found) == len(ranks) and all(r <= 10 for r in found))
    return m
