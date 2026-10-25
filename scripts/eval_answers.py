"""End-to-end answer evaluation with an LLM judge.

For each question and config: retrieve, generate a cited answer, then a judge model scores
  correctness   vs the reference answer (correct / partial / incorrect)
  faithfulness  share of the answer's claims supported by the excerpts it cites
  abstention    unanswerable questions should be declined; answerable ones should not

Answerer and judge are configurable; scripts/validate_judge.py checks a local judge against Gemini.
Usage: python scripts/eval_answers.py --answer-provider ollama --judge-provider ollama [--limit N]
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag.evaluation import EVAL, load_testset
from src.rag.llm import get_llm, parse_json
from src.rag.pipeline import SYSTEM, SYSTEM_V1, RAGPipeline, format_sources
from src.rag.retriever import Retriever

# name -> (search kwargs, system prompt, query decomposition)
CONFIGS = {
    "baseline": (dict(mode="bm25", rerank=False), SYSTEM_V1, False),
    "full": (dict(mode="hybrid", rerank=True), SYSTEM_V1, False),
    "full_premise": (dict(mode="hybrid", rerank=True), SYSTEM, False),
    # Query decomposition is wired in but excluded by default: it did not change retrieval (REPORT.md §4).
    "full_premise_decomp": (dict(mode="hybrid", rerank=True), SYSTEM, True),
}

JUDGE_SYSTEM = "You are a strict, fair evaluator of a question-answering system. Reply with JSON only."

JUDGE_PROMPT = """Question: {question}

Reference answer: {reference}

Numbered excerpts the system was given:
{sources}

System answer:
{answer}

Evaluate the system answer.
1. "correctness": compare with the reference answer. "correct" if it contains the reference's key \
facts (extra correct detail is fine), "partial" if it has some but misses or blurs key facts, \
"incorrect" if it misses them, contradicts them, or declines to answer.{unanswerable_note}
2. "claims": split the answer into its factual claims. For each, "supported": true only if the \
excerpts it cites (the [n] markers) state or directly imply it. A claim with no citation is unsupported.
3. "declined": true if the answer says it cannot answer / the information is not available.

Format: {{"correctness": "correct|partial|incorrect", "declined": true|false, \
"claims": [{{"claim": "...", "supported": true|false}}], "reason": "one sentence"}}"""

UNANSWERABLE_NOTE = """
   This question is NOT answerable from the corpus: "correct" means the system declined; \
"incorrect" means it gave a specific answer anyway."""


def judge(judge_llm, q, out):
    prompt = JUDGE_PROMPT.format(
        question=q["question"], reference=q["reference_answer"],
        sources=format_sources(out["sources"]) if out["sources"] else "(none)",
        answer=out["answer"],
        unanswerable_note=UNANSWERABLE_NOTE if q["type"] == "unanswerable" else "")
    return parse_json(judge_llm.generate(JUDGE_SYSTEM, prompt))


def parse_claims(judgment: dict) -> list[dict]:
    """Judges sometimes return claims as plain strings; those carry no verdict and are skipped."""
    claims = judgment.get("claims") or []
    return [{"claim": str(c.get("claim", "")), "supported": bool(c.get("supported"))}
            for c in claims if isinstance(c, dict) and "supported" in c]


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0}
    ans = df[df.type != "unanswerable"]
    una = df[df.type == "unanswerable"]
    score = ans.correctness.map({"correct": 1.0, "partial": 0.5, "incorrect": 0.0})
    with_claims = ans[ans.n_claims > 0]
    return {
        "n": len(df),
        "accuracy (correct)": round((ans.correctness == "correct").mean(), 3),
        "accuracy (partial=0.5)": round(score.mean(), 3),
        "faithfulness (claims supported)": round(with_claims.n_supported.sum() / max(with_claims.n_claims.sum(), 1), 3),
        "hallucination rate (>=1 unsupported claim)": round((with_claims.n_supported < with_claims.n_claims).mean(), 3),
        "false abstention (answerable)": round(ans.declined.mean(), 3),
        "correct abstention (unanswerable)": round(una.declined.mean(), 3) if len(una) else None,
        "single acc": round((ans[ans.type == "single"].correctness == "correct").mean(), 3),
        "multi acc": round((ans[ans.type == "multi"].correctness == "correct").mean(), 3),
    }


def main(config_names: list[str], limit: int | None, answer_provider=None, answer_model=None,
         judge_provider=None, judge_model=None, workers: int = 3):
    load_dotenv()
    items = load_testset()[:limit] if limit else load_testset()
    answer_llm = get_llm(answer_provider, answer_model)
    judge_llm = get_llm(judge_provider, judge_model)
    retriever = Retriever()
    rag = RAGPipeline(retriever, answer_llm)
    out_dir = EVAL / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(items)} questions | answerer {answer_llm.name} | judge {judge_llm.name}")

    summaries = {}
    for name in config_names:
        # Retrieval runs serially on the GPU; LLM calls run in parallel threads.
        search_kw, system, decompose = CONFIGS[name]
        rag.system = system
        hits = [rag.retrieve(q["question"], k=8, decompose=decompose, **search_kw)[1]
                for q in tqdm(items, desc=f"{name}: retrieve")]

        def run(args):
            q, h = args
            try:
                out = rag.generate(q["question"], h)
                j = judge(judge_llm, q, out)
                claims = parse_claims(j)
                return {
                    "id": q["id"], "type": q["type"], "question": q["question"],
                    "reference": q["reference_answer"], "answer": out["answer"],
                    "correctness": j.get("correctness"), "declined": bool(j.get("declined")),
                    "n_claims": len(claims), "n_supported": sum(c["supported"] for c in claims),
                    "unsupported": [c["claim"] for c in claims if not c["supported"]],
                    "reason": j.get("reason"), "gold_doc_ids": q["gold_doc_ids"],
                    "retrieved_doc_ids": [x["doc_id"] for x in h],
                    "retrieved_chunk_ids": [x["chunk_id"] for x in h], "cited": out["cited"],
                }
            except Exception as e:  # noqa: BLE001 - record and continue
                return {"id": q["id"], "type": q["type"], "error": str(e)[:300]}

        with ThreadPoolExecutor(workers) as pool:
            rows = list(tqdm(pool.map(run, zip(items, hits)), total=len(items), desc=f"{name}: answer+judge"))
        errors = [r for r in rows if "error" in r]
        df = pd.DataFrame([r for r in rows if "error" not in r])
        with open(out_dir / f"answers_{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summaries[name] = {**summarize(df), "errors": len(errors)}
        print(name, json.dumps(summaries[name], indent=1))

    table = pd.DataFrame(summaries)
    table.to_csv(out_dir / "answers_summary.csv")
    (out_dir / "answers_summary.md").write_text(
        f"Answerer: {answer_llm.name} | Judge: {judge_llm.name}\n\n" + table.to_markdown(), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", default=["baseline", "full", "full_premise"], choices=list(CONFIGS))
    p.add_argument("--limit", type=int)
    p.add_argument("--answer-provider")
    p.add_argument("--answer-model")
    p.add_argument("--judge-provider")
    p.add_argument("--judge-model")
    p.add_argument("--workers", type=int, default=3)
    a = p.parse_args()
    main(a.configs, a.limit, a.answer_provider, a.answer_model, a.judge_provider, a.judge_model, a.workers)
