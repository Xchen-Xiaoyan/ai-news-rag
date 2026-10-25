# Evaluation report

System: hybrid RAG over 33,234 deduplicated AI-industry news articles (2022-01 to 2026-02),
188,454 chunks. Everything runs locally at $0 on an RTX 5060 Laptop (8 GB): bge embeddings and
reranker, `qwen2.5:7b` via Ollama for generation and judging, with a free-tier Gemini model as a
stronger reference judge on a sample.

All numbers below are on the **reviewed test set (75 questions)** unless marked "pre-review".

## 1. Test set

Questions were generated from the 3,024 articles that carry labels from the original topic-modeling
project, passed through automatic quality gates, then reviewed question by question.

| Type | Gold label | Draft | After review |
|---|---|---|---|
| single: answerable from one passage | that article (any of several, if they cover the same event) | 60 | 50 |
| multi: needs facts from two articles | both articles | 16 | 9 |
| unanswerable: post-2026 events, unpublished internal figures, fictional companies | should abstain | 20 | 16 |

**Automatic gates** (`scripts/filter_testset.py`, 207 candidates → 96 draft):

| Stage | Dropped | Rule |
|---|---|---|
| rules | 18 | mentions "the article/passage", too short, duplicate |
| critic | 50 | LLM check: no concrete entity in the question, generic enough that many articles answer it, reference not in the gold text, or a "multi" question answerable from one source |
| probe | 6 | "unanswerable" questions searched against the real index; dropped if the retrieved excerpts answer them |

Asking the 7B critic for a 1-5 specificity score failed: it rated 130/130 questions "3". Binary checks
with examples fixed this.

**Review** (96 → 75; every decision and reason is in `testset_review.csv`). The review was done by
an AI assistant (Claude) reading each question against its gold passage. It is not an independent
human annotation.

| Problem | Count | Action |
|---|---|---|
| Same article as a Gemini-written question (the two generators sampled the same docs) | 9 | drop duplicate |
| Answer comes from a page sidebar ("trending" list), not the article | 2 | drop |
| Event-logistics trivia | 1 | drop |
| "Multi" question where one gold article isn't actually needed, or the link is contrived | 5 | drop |
| "Multi" question whose two articles cover the same event | 2 | relabel as single (either article counts) |
| "Unanswerable" forecast question ("By 2028, how will…") that real forecasts in the archive may answer | 4 | drop |
| Ambiguous wording (e.g. "the day of the outage" when ChatGPT had many) | 3 | rewrite |

Remaining caveats:
- LLM-written questions keep lexical overlap with their source passage, which favors BM25.
- Most unanswerable questions follow two templates ("internal cost per query", "in 2027/2028").
- n = 9 multi questions is too few for strong conclusions.

## 2. Retrieval (59 answerable questions, doc-level)

| Config | Hit@1 | Hit@5 | Hit@10 | MRR | multi: both docs in top 10 | latency |
|---|---|---|---|---|---|---|
| BM25 | 0.729 | **0.949** | 0.966 | 0.824 | **0.78** | 6 ms |
| Dense (bge-small) | 0.661 | 0.881 | 0.881 | 0.743 | 0.78 | 23 ms |
| Hybrid (RRF) | 0.695 | 0.949 | **0.983** | 0.807 | 0.67 | 19 ms |
| BM25 + rerank | 0.797 | 0.932 | 0.983 | 0.866 | 0.78 | 0.5 s |
| **Hybrid + rerank (default)** | 0.797 | 0.932 | 0.983 | 0.866 | 0.67 | 0.4 s |
| Hybrid + rerank, 100-word chunks | **0.831** | **0.966** | 0.983 | **0.890** | 0.56 | 0.2 s |
| Hybrid + rerank, 350-word chunks | 0.797 | 0.915 | 0.949 | 0.852 | 0.67 | 0.3 s |
| Dense (bge-base) | 0.627 | 0.898 | 0.949 | 0.739 | 0.56 | 19 ms |
| Hybrid + rerank (bge-base) | 0.797 | 0.932 | 0.966 | 0.864 | 0.67 | 0.3 s |
| Hybrid + rerank + query decomposition | 0.797 | 0.932 | 0.983 | 0.864 | 0.67 | 0.9 s |

Significance testing used a paired bootstrap with 10,000 resamples against BM25 (`retrieval_significance.md`):

- **Dense-only retrieval is significantly worse than BM25** (Hit@10 −8.5 pp, CI [−15.3, −1.7]),
  and bge-base doesn't fix it (MRR −8.5 pp, significant). On this news corpus, keyword
  overlap carries most of the signal. Dense vectors are useful as a complement, not a replacement.
- The reranker's gain is consistent across every first stage: Hit@1 +6.8 pp over BM25, and +10.2 pp over
  hybrid without rerank. It is not significant at n = 59.
- The pre-review run (76 questions) found hybrid's Hit@10 gain significant (+5.3 pp). On the reviewed set
  BM25 already reaches 0.966, so the gain shrinks to +1.7 pp and is no longer significant. Several of
  the questions hybrid had "rescued" were bad questions.
- Query decomposition: no change on any metric, at 2× latency. Details in §4.

## 3. Answers (75 questions; answerer and judge qwen2.5:7b)

| Metric | Baseline (BM25, prompt v1) | Full (hybrid + rerank, v1) | **Full + premise check (v2)** |
|---|---|---|---|
| Accuracy (correct), local judge | 1.000 | 0.949 | 0.898 |
| single / multi accuracy | 1.00 / 1.00 | 0.98 / 0.78 | 0.96 / 0.56 |
| False abstention (answerable) | 0.00 | 0.00 | 0.00 |
| **Correct abstention (unanswerable)** | 0.875 | 0.812 | **1.000** |

Local-judge accuracy is close to saturation and is lenient (next section), so it can't separate the
retrievers. The informative numbers are abstention, which the judge scores reliably, and the
stronger-judge sample below.

### Is the local judge trustworthy?

30 pre-review "full" answers were re-judged by `gemini-3.8-flash` (`scripts/validate_judge.py`):

| Judgment | Agreement | Cohen's kappa | Verdict |
|---|---|---|---|
| declined to answer | 0.97 | 0.87 | reliable: the abstention numbers stand |
| correctness | 0.83 | 0.44 | lenient: 5 answers it called "correct" Gemini rated "partial" |
| any unsupported claim | 0.50 | 0.10 | **unreliable** |

On those 30 answers:

| | Local judge | Gemini judge |
|---|---|---|
| Accuracy (correct) | 0.90 | **0.77** |
| Faithfulness (claims supported by cited excerpts) | 0.95 | **0.70** |
| Answers with ≥ 1 unsupported claim | 0.07 | **0.57** |

Most claims Gemini flags are the 7B answerer's own summarizing sentences ("Klarna's experience
highlights…"), but some are real errors (a made-up "by 2028" target). Faithfulness numbers must come from the
stronger judge. Gemini re-judging of v2 stopped at 5/30 when the free-tier daily quota ran out.

## 4. Failure analysis and iterations

1. **False-premise questions.** v1 answered "How did the 2027 export controls affect Nvidia?" by
   restating 2023 facts as if they were about 2027. Prompt v2 adds a premise check: if the asked-about
   period is later than every excerpt, or the event is never mentioned, it declines and names the closest
   related information. **Correct abstention went from 0.81 to 1.00, with no false abstentions.** v2 ships.
2. **v2's cost is multi-document questions**: accuracy 0.78 → 0.56 on n = 9, i.e. 2 questions. v2 declines
   when retrieval surfaced only one of the two needed articles, a gap v1 papered over with a
   one-sided answer. This is the trade-off v2 is meant to make, but it needs a larger multi set to quantify.
3. **Ungrounded synthesis.** The 7B answerer adds interpretive sentences the excerpts don't state
   (Gemini faithfulness 0.70). The demo uses Gemini as the answerer.
4. **Query decomposition: tried, no gain, not shipped.** The LLM splits multi-entity questions into
   ≤ 3 sub-queries, retrieves for each plus the original, and interleaves the lists
   (`RAGPipeline.retrieve(decompose=True)`). No retrieval metric changed. On the pre-review set, ~4 of
   the 7 multi "misses" had retrieved *another article on the same event*. Review relabeled those
   questions, and multi both-docs@10 for BM25 rose from 0.50 to 0.78. So the original gap was mostly
   labeling, which decomposition can't fix. It stays opt-in.
5. **Infrastructure:** Ollama 0.40 on the RTX 5060 sometimes entered a bad state and emitted "8888…"
   until it aborted with HTTP 500 ("token repeat limit"). The fix was to detect degenerate output, unload the model, and
   retry. It ran single-threaded.

## 5. Serving on a free CPU (2 vCPUs)

Per-stage latency on CPU with 2 threads: BM25 < 1 ms and dense 37 ms. The reranker is everything.

| Reranker (20 candidates unless noted) | Hit@1 | Hit@5 | MRR | Rerank latency (2 vCPU) |
|---|---|---|---|---|
| bge-reranker-base, 512 tokens (GPU default) | **0.797** | **0.983** | **0.867** | 10.7 s |
| bge-reranker-base, 50 candidates | 0.797 | 0.932 | 0.866 | ~27 s |
| bge-reranker-base truncated to 256 tokens | 0.712 | 0.949 | 0.812 | 6.6 s |
| bge-reranker-base, 256 tokens, 10 candidates | 0.712 | 0.949 | 0.818 | 3.8 s |
| **ms-marco-MiniLM-L-6 (CPU default)** | 0.763 | 0.915 | 0.832 | **1.5 s** |

- 20 candidates score as well as 50, so the demo reranks 20. The eval configs above used 50.
- Truncating passages costs more accuracy than switching to a reranker 12× smaller.
- The hosted demo uses MiniLM (−3.4 pp Hit@1, still above BM25's 0.729) for about 2.8 s end to end.

## 6. Next steps

- A larger, hand-written multi-document and unanswerable set, to break the templates and reach n ≥ 30 per type.
- Complete the Gemini re-judging of v2 (cached calls resume). Try a "no uncited sentences" rule and
  measure faithfulness.
- Try 100-word chunks as the default (best Hit@1 / MRR, not yet significant).

## Reproduce

```bash
python scripts/build_testset.py --provider ollama --single 120 --multi 40 --unanswerable 40
python scripts/filter_testset.py          # draft + review sheet; review decisions live in the CSV
python scripts/eval_retrieval.py && python scripts/significance.py
python scripts/eval_answers.py --answer-provider ollama --judge-provider ollama --workers 1
python scripts/validate_judge.py --config full --n 30 --model gemini-3.8-flash
```
