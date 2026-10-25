"""Quality-gate generated questions, then build the draft test set and the human review sheet.

Filters, in order:
  1. rules     drop questions that mention "the article/passage/sources", are too short, or duplicate
  2. critic    an LLM rates each answerable question: is it specific (needs THIS article) and is the
               reference answer supported by the gold text? multi questions must need both sources
  3. probe     unanswerable questions are searched against the real index; if the retrieved excerpts
               do answer them, they are not unanswerable and are dropped

Outputs eval/testset_draft.jsonl, eval/testset_review.csv and eval/filter_log.csv.
"""
import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag import config
from src.rag.evaluation import EVAL
from src.rag.llm import get_llm, parse_json
from src.rag.pipeline import format_sources

SOURCE_REF = re.compile(r"\b(the|this|both|these) (article|passage|text|report|sources?|excerpts?)\b|"
                        r"\bsource [ab]\b|\bbased on the information\b", re.I)

CRITIC_PROMPT = """You review questions for a test set of a news-archive question-answering system.

Question: {question}
Reference answer: {answer}

Gold source text:
{gold}

Answer these yes/no checks:
- "names_entity": does the QUESTION itself name a concrete company, product, person, organization, place, figure or dated event? ("Why did Klarna rehire human agents?" -> true; "How can AI improve marketing copy?" -> false)
- "generic": could many unrelated AI news articles answer it equally well? ("What are the benefits of AI in healthcare?" -> true; "What CAGR does the 2022 report project for CAD breast-screening software?" -> false)
- "supported": is the reference answer actually stated in the gold text?
- "needs_all_sources": if there are 2 sources, does a full answer need facts from both? (true if only 1 source)

Reply with JSON only: {{"names_entity": true, "generic": false, "supported": true, "needs_all_sources": true, "reason": "one short sentence"}}"""

PROBE_PROMPT = """Question: {question}

Excerpts from a news archive:
{sources}

Do these excerpts contain the specific information needed to answer the question (fully or mostly)? \
Reply with JSON only: {{"answerable": true|false, "reason": "one short sentence"}}"""


def norm(q: str) -> str:
    return re.sub(r"[^a-z0-9]", "", q.lower())


def main(quota: dict, provider: str | None, model: str | None):
    load_dotenv()
    llm = get_llm(provider, model)
    raw = []
    # Gemini-written questions first: they were higher quality and win ties/dedup.
    for name in ["testset_raw_gemini.jsonl", "testset_raw_ollama.jsonl"]:
        p = EVAL / name
        if p.exists():
            for line in open(p, encoding="utf-8"):
                r = json.loads(line)
                r.setdefault("generator", "gemini/gemini-3.8-flash" if "gemini" in name else "ollama")
                raw.append(r)
    print(f"raw candidates: {len(raw)}  {pd.Series([r['type'] for r in raw]).value_counts().to_dict()}")

    log, seen, cands = [], set(), []
    for r in raw:
        q = r["question"]
        reason = ("mentions source" if SOURCE_REF.search(q) else "too short" if len(q.split()) < 8
                  else "duplicate" if norm(q) in seen else None)
        seen.add(norm(q))
        if reason:
            log.append({"question": q, "type": r["type"], "stage": "rules", "reason": reason})
        else:
            cands.append(r)

    chunks = pd.read_parquet(config.CHUNKS).set_index("chunk_id")

    def critic(r):
        gold = "\n\n".join(f"[Source {i + 1}] {chunks.loc[c].text}" for i, c in enumerate(r["gold_chunk_ids"]))
        try:
            j = parse_json(llm.generate("You are a strict reviewer. Reply with JSON only.", CRITIC_PROMPT.format(
                question=r["question"], answer=r["reference_answer"], gold=gold)))
        except Exception as e:  # noqa: BLE001
            return r, False, f"critic error: {e}"[:150]
        r["critic"] = {k: bool(j.get(k)) for k in ["names_entity", "generic", "supported", "needs_all_sources"]}
        r["critic"]["reason"] = j.get("reason", "")
        if not j.get("names_entity") or j.get("generic"):
            return r, False, f"not specific: {j.get('reason', '')}"
        if not j.get("supported"):
            return r, False, f"reference unsupported: {j.get('reason', '')}"
        if r["type"] == "multi" and not j.get("needs_all_sources"):
            return r, False, f"multi answerable from one source: {j.get('reason', '')}"
        return r, True, ""

    answerable = [r for r in cands if r["type"] != "unanswerable"]
    with ThreadPoolExecutor(3) as pool:
        judged = list(tqdm(pool.map(critic, answerable), total=len(answerable), desc="critic"))

    from src.rag.retriever import Retriever
    retriever = Retriever()

    def probe(r, hits):
        try:
            j = parse_json(llm.generate("You are a careful fact checker. Reply with JSON only.", PROBE_PROMPT.format(
                question=r["question"], sources=format_sources(hits))))
        except Exception as e:  # noqa: BLE001
            return r, False, f"probe error: {e}"[:150]
        if j.get("answerable"):
            return r, False, f"corpus can answer it: {j.get('reason', '')}"
        return r, True, ""

    unans = [r for r in cands if r["type"] == "unanswerable"]
    hits = [retriever.search(r["question"], k=5) for r in tqdm(unans, desc="probe: retrieve")]
    with ThreadPoolExecutor(3) as pool:
        probed = list(tqdm(pool.map(lambda a: probe(*a), zip(unans, hits)), total=len(unans), desc="probe"))

    kept = {"single": [], "multi": [], "unanswerable": []}
    for stage, results in [("critic", judged), ("probe", probed)]:
        for r, ok, reason in results:
            if ok:
                kept[r["type"]].append(r)
            else:
                log.append({"question": r["question"], "type": r["type"], "stage": stage, "reason": reason})

    final = []
    for kind, n in quota.items():
        final += kept[kind][:n]
        print(f"{kind}: {len(kept[kind])} passed, keeping {min(n, len(kept[kind]))}")
    for i, r in enumerate(final):
        r["id"] = f"q{i:03d}"

    with open(EVAL / "testset_draft.jsonl", "w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pd.DataFrame([{
        "id": r["id"], "type": r["type"], "keep": "", "question": r["question"],
        "reference_answer": r["reference_answer"], "gold_titles": " || ".join(r.get("gold_titles", [])),
        "gold_dates": " || ".join(r.get("gold_dates", [])), "generator": r["generator"],
         "notes": ""} for r in final]
    ).to_csv(EVAL / "testset_review.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(log).to_csv(EVAL / "filter_log.csv", index=False, encoding="utf-8-sig")
    print(f"dropped {len(log)}: {pd.DataFrame(log).groupby(['stage', 'type']).size().to_dict()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--single", type=int, default=60)
    p.add_argument("--multi", type=int, default=20)
    p.add_argument("--unanswerable", type=int, default=20)
    p.add_argument("--provider", default="ollama")
    p.add_argument("--model")
    a = p.parse_args()
    main({"single": a.single, "multi": a.multi, "unanswerable": a.unanswerable}, a.provider, a.model)
