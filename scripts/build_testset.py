"""Generate a draft evaluation set from the labeled (high_conf) articles.

Three question types, each with gold labels for scoring:
  single        answerable from one chunk           gold = that article
  multi         needs two articles on one topic      gold = both articles
  unanswerable  plausible but not in the corpus      gold = abstain

Outputs eval/testset_raw_<provider>.jsonl; scripts/filter_testset.py then builds the draft + review sheet.
Usage: python scripts/build_testset.py [--single 60 --multi 20 --unanswerable 20]
"""
import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.rag import config
from src.rag.llm import get_llm, parse_json

EVAL = ROOT / "eval"

COMMON_RULES = """- The question must be self-contained: name the companies, products, places or time frame. \
Never say "the article", "the passage" or "this report".
- Paraphrase: do not copy distinctive phrases from the text, so that keyword matching alone is not enough.
- Ask about substance (what happened, why, how much, what effect on an industry), not trivia like author names.
- Be specific enough that THIS text is needed: include a concrete detail (a named company, product, \
figure or event). Avoid generic questions like "How can AI improve marketing?" that many articles answer.
- Reply with JSON only."""

SINGLE_PROMPT = """You are building a test set for a question-answering system over an archive of AI industry news.

Article title: {title}
Date: {date}
Passage:
{text}

Write ONE question that this passage answers completely, and the reference answer (1-2 sentences, \
based only on the passage).
{rules}
If the passage has no substantive, checkable fact, reply {{"skip": true}}.
Format: {{"question": "...", "answer": "..."}}"""

MULTI_PROMPT = """You are building a test set for a question-answering system over an archive of AI industry news.

Source A - {title_a} ({date_a}):
{text_a}

Source B - {title_b} ({date_b}):
{text_b}

Write ONE question whose full answer needs facts from BOTH sources (e.g. compare two companies' \
approaches, or how a situation changed between the two dates), and the reference answer (2-3 sentences, \
based only on the two sources).
{rules}
If the two sources share no meaningful link, reply {{"skip": true}}.
Format: {{"question": "...", "answer": "..."}}"""

UNANSWERABLE_PROMPT = """You are building a test set for a question-answering system over an archive of \
AI industry news published between January 2022 and February 2026.

Here is an article from the archive, for topic inspiration only:
Title: {title} ({date})
{text}

Write ONE realistic question on the same topic that the archive almost certainly CANNOT answer. Use this strategy: {strategy}
The question should sound like something a real analyst would ask, and be specific enough that a \
system without the facts would be tempted to make up an answer.
{rules}
Format: {{"question": "...", "answer": "Not answerable from the corpus."}}"""

STRATEGIES = [
    "ask about an event or figure from 2027 or later (after the archive ends)",
    "ask for a precise internal figure the company would never publish (e.g. exact internal cost per query, "
    "a named employee's salary, a confidential contract value)",
]


def load():
    docs = pd.read_parquet(config.DOCS)
    chunks = pd.read_parquet(config.CHUNKS)
    docs = docs[docs.in_high_conf].copy()
    docs["primary_industry"] = docs.industries.map(lambda xs: xs[0] if xs is not None and len(xs) else "other")
    chunks = chunks[chunks.doc_id.isin(docs.doc_id) & (chunks.text.str.split().str.len() >= 150)]
    return docs.set_index("doc_id"), chunks


def stratified_docs(docs: pd.DataFrame, n: int, rng: random.Random) -> list[str]:
    """Round-robin over industries so the test set isn't dominated by one sector."""
    groups = {k: list(v) for k, v in docs.groupby("primary_industry").groups.items()}
    for g in groups.values():
        rng.shuffle(g)
    out, keys = [], sorted(groups)
    while len(out) < n and any(groups.values()):
        for k in keys:
            if groups[k] and len(out) < n:
                out.append(groups[k].pop())
    return out


def short_title(t: str) -> str:
    return t.split(" | ")[0][:150]


def main(n_single: int, n_multi: int, n_unans: int, provider: str | None = None, model: str | None = None,
         seed: int = 42):
    load_dotenv()
    rng = random.Random(seed)
    llm = get_llm(provider, model)
    docs, chunks = load()
    by_doc = chunks.groupby("doc_id")
    print(f"labeled docs with usable chunks: {chunks.doc_id.nunique():,}")

    def pick_chunk(doc_id):
        g = by_doc.get_group(doc_id)
        return g.iloc[rng.randrange(len(g))]

    jobs = []
    eligible = docs.loc[docs.index.isin(chunks.doc_id)]
    # Over-sample by 40%: the model is told to skip weak passages.
    for doc_id in stratified_docs(eligible, int(n_single * 1.4), rng):
        c, d = pick_chunk(doc_id), docs.loc[doc_id]
        jobs.append(("single", [doc_id], [c.chunk_id], SINGLE_PROMPT.format(
            title=short_title(d.title), date=d.date.date(), text=c.text, rules=COMMON_RULES)))

    labeled = eligible[eligible.topic_label.notna()]
    topics = [t for t, g in labeled.groupby("topic_label") if g.source.nunique() >= 2]
    for _ in range(int(n_multi * 1.4)):
        g = labeled[labeled.topic_label == rng.choice(topics)]
        a = rng.choice(list(g.index))
        rest = g[g.source != g.loc[a].source]
        b = rng.choice(list(rest.index))
        ca, cb = pick_chunk(a), pick_chunk(b)
        jobs.append(("multi", [a, b], [ca.chunk_id, cb.chunk_id], MULTI_PROMPT.format(
            title_a=short_title(docs.loc[a].title), date_a=docs.loc[a].date.date(), text_a=ca.text,
            title_b=short_title(docs.loc[b].title), date_b=docs.loc[b].date.date(), text_b=cb.text,
            rules=COMMON_RULES)))

    for i, doc_id in enumerate(stratified_docs(eligible, n_unans, rng)):
        c, d = pick_chunk(doc_id), docs.loc[doc_id]
        jobs.append(("unanswerable", [], [], UNANSWERABLE_PROMPT.format(
            title=short_title(d.title), date=d.date.date(), text=c.text,
            strategy=STRATEGIES[i % len(STRATEGIES)], rules=COMMON_RULES)))

    def run(job):
        kind, doc_ids, chunk_ids, prompt = job
        try:
            out = parse_json(llm.generate("You write evaluation data. Reply with JSON only.", prompt))
        except Exception as e:  # noqa: BLE001 - one bad generation shouldn't stop the batch
            return None, f"{kind}: {e}"
        if not isinstance(out, dict) or out.get("skip"):
            return None, None
        question, answer = out.get("question"), out.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str) or not question.strip():
            return None, f"{kind}: malformed reply {str(out)[:120]}"
        return {"type": kind, "question": question.strip(), "reference_answer": answer.strip(),
                "gold_doc_ids": doc_ids, "gold_chunk_ids": chunk_ids}, None

    with ThreadPoolExecutor(3) as pool:
        results = list(tqdm(pool.map(run, jobs), total=len(jobs), desc="generating"))
    errors = [e for _, e in results if e]
    items = [r for r, _ in results if r]

    quota = {"single": n_single, "multi": n_multi, "unanswerable": n_unans}
    kept = []
    for kind, n in quota.items():
        kept += [r for r in items if r["type"] == kind][:n]
    for i, r in enumerate(kept):
        r["id"] = f"q{i:03d}"
        r["gold_titles"] = [short_title(docs.loc[d].title) for d in r["gold_doc_ids"]]
        r["gold_dates"] = [str(docs.loc[d].date.date()) for d in r["gold_doc_ids"]]
        r["industry"] = docs.loc[r["gold_doc_ids"][0]].primary_industry if r["gold_doc_ids"] else ""
        r["generator"] = llm.name

    EVAL.mkdir(exist_ok=True)
    out_path = EVAL / f"testset_raw_{llm.name.split('/')[0]}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(pd.Series([r["type"] for r in kept]).value_counts().to_dict(), f"errors: {len(errors)}")
    for e in errors[:5]:
        print("  ", e)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--single", type=int, default=60)
    p.add_argument("--multi", type=int, default=20)
    p.add_argument("--unanswerable", type=int, default=20)
    p.add_argument("--provider", help="gemini | ollama (default: $LLM_PROVIDER)")
    p.add_argument("--model")
    a = p.parse_args()
    main(a.single, a.multi, a.unanswerable, a.provider, a.model)
