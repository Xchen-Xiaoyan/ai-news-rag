"""Paired bootstrap on retrieval results: is config A really better than config B on this test set?

Reads eval/results/retrieval_details.jsonl, writes eval/results/retrieval_significance.md.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag.evaluation import EVAL

BASELINE = "bm25 | w200 | bge-small-en-v1.5"
N_BOOT = 10_000


def per_question(details: pd.DataFrame) -> pd.DataFrame:
    def first_rank(ranks):
        found = [r for r in ranks if r is not None]
        return min(found) if found else None
    d = details.copy()
    d["first"] = d.ranks.map(first_rank)
    d["hit@1"] = (d["first"] == 1).astype(float)
    d["hit@10"] = (d["first"] <= 10).astype(float)
    d["mrr"] = d["first"].map(lambda r: 1.0 / r if pd.notna(r) else 0.0)
    return d


def main():
    details = pd.read_json(EVAL / "results" / "retrieval_details.jsonl", lines=True)
    d = per_question(details)
    rng = np.random.default_rng(0)
    ids = d.id.unique()
    boot = rng.integers(0, len(ids), size=(N_BOOT, len(ids)))
    rows = []
    for metric in ["hit@1", "hit@10", "mrr"]:
        wide = d.pivot(index="id", columns="config", values=metric).loc[ids]
        base = wide[BASELINE].to_numpy()
        for cfg in wide.columns:
            x = wide[cfg].to_numpy()
            means = x[boot].mean(axis=1)
            diffs = (x - base)[boot].mean(axis=1)
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            rows.append({"metric": metric, "config": cfg, "mean": round(x.mean(), 3),
                         "95% CI": f"[{np.percentile(means, 2.5):.3f}, {np.percentile(means, 97.5):.3f}]",
                         "diff vs bm25": round((x - base).mean(), 3), "diff 95% CI": f"[{lo:.3f}, {hi:.3f}]",
                         "significant": "yes" if lo > 0 or hi < 0 else "no"})
    out = pd.DataFrame(rows)
    md = (f"Paired bootstrap ({N_BOOT:,} resamples, n = {len(ids)} answerable questions). "
          f"Baseline: {BASELINE}.\n\n" + out.to_markdown(index=False))
    (EVAL / "results" / "retrieval_significance.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
