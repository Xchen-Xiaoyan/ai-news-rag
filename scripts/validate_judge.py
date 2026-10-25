"""How far can we trust the local judge? Re-judge a sample of answers with a stronger model and
report agreement (raw % and Cohen's kappa).

Usage: python scripts/validate_judge.py --config full --n 30 --provider gemini --model gemini-3.5-flash-lite
"""
import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag import config
from src.rag.evaluation import EVAL, load_testset
from src.rag.llm import get_llm
from scripts.eval_answers import judge, parse_claims


def cohen_kappa(a: list, b: list) -> float:
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main(cfg: str, n: int, provider: str, model: str | None, seed: int = 0):
    load_dotenv()
    rows = [json.loads(l) for l in open(EVAL / "results" / f"answers_{cfg}.jsonl", encoding="utf-8")]
    rows = [r for r in rows if "error" not in r]
    questions = {q["id"]: q for q in load_testset()}
    random.Random(seed).shuffle(rows)
    rows = rows[:n]

    chunks = pd.read_parquet(config.CHUNKS).set_index("chunk_id")
    docs = pd.read_parquet(config.DOCS, columns=["doc_id", "title", "source", "date"]).set_index("doc_id")

    def sources(chunk_ids):
        out = []
        for cid in chunk_ids:
            c = chunks.loc[cid]
            d = docs.loc[c.doc_id]
            out.append({"title": d.title, "source": d.source, "date": d.date.strftime("%Y-%m-%d"), "text": c.text})
        return out

    strong = get_llm(provider, model)
    print(f"re-judging {len(rows)} answers from config '{cfg}' with {strong.name}")
    pairs = []
    for r in tqdm(rows):
        try:
            j = judge(strong, questions[r["id"]], {"answer": r["answer"], "sources": sources(r["retrieved_chunk_ids"])})
        except Exception as e:  # noqa: BLE001
            print("  skip", r["id"], str(e)[:120])
            continue
        claims = parse_claims(j)
        pairs.append({
            "id": r["id"], "type": r["type"],
            "local_correctness": r["correctness"], "strong_correctness": j.get("correctness"),
            "local_declined": r["declined"], "strong_declined": bool(j.get("declined")),
            "local_hallucinated": r["n_supported"] < r["n_claims"],
            "strong_hallucinated": sum(c["supported"] for c in claims) < len(claims),
            "local_claims": r["n_claims"], "local_supported": r["n_supported"],
            "strong_claims": len(claims), "strong_supported": sum(c["supported"] for c in claims),
            "strong_unsupported": " || ".join(c["claim"] for c in claims if not c["supported"]),
        })

    df = pd.DataFrame(pairs)
    report = {"n": len(df), "strong judge": strong.name}
    for field in ["correctness", "declined", "hallucinated"]:
        a, b = df[f"local_{field}"].tolist(), df[f"strong_{field}"].tolist()
        report[f"{field} agreement"] = round(sum(x == y for x, y in zip(a, b)) / len(a), 3)
        report[f"{field} kappa"] = round(cohen_kappa(a, b), 3)
    for who in ["local", "strong"]:
        report[f"{who} faithfulness (claims supported)"] = round(
            df[f"{who}_supported"].sum() / max(df[f"{who}_claims"].sum(), 1), 3)
        report[f"{who} hallucination rate"] = round(df[f"{who}_hallucinated"].mean(), 3)
    report[f"strong accuracy (correct)"] = round((df.strong_correctness == "correct").mean(), 3)
    report[f"local accuracy (correct)"] = round((df.local_correctness == "correct").mean(), 3)
    print(json.dumps(report, indent=1))
    df.to_csv(EVAL / "results" / f"judge_validation_{cfg}.csv", index=False)
    (EVAL / "results" / f"judge_validation_{cfg}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="full")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--provider", default="gemini")
    p.add_argument("--model", default="gemini-3.5-flash-lite")
    a = p.parse_args()
    main(a.config, a.n, a.provider, a.model)
