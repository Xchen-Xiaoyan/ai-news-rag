# Evaluation report (Phase 2)

System: hybrid RAG over 33,234 deduplicated AI-industry news articles (2022-01 to 2026-02),
188,454 chunks. Everything runs locally at $0 on an RTX 5060 Laptop (8 GB): bge embeddings and
reranker, `qwen2.5:7b` via Ollama for generation and judging, with a free-tier Gemini model as a
stronger reference judge on a sample.

## 1. Test set

96 questions, generated from the 3,024 articles that carry labels from the original topic-modeling
project, then passed through automatic quality gates (`scripts/filter_testset.py`).

| Type | Gold label | Kept |
|---|---|---|
| single: answerable from one passage | that article | 60 |
| multi: needs facts from two articles on one topic | both articles | 16 |
| unanswerable: plausible, but not in the archive (post-2026 events, unpublished internal figures) | should abstain | 20 |

Filtering funnel (207 raw candidates; 20 from Gemini, 187 from qwen2.5:7b):

| Stage | Dropped | Rule |
|---|---|---|
| rules | 18 | mentions "the article/passage", too short, duplicate |
| critic | 50 | LLM check: question names no concrete entity, or is generic enough that many articles answer it, or the reference answer is not in the gold text, or a "multi" question is answerable from one source |
| probe | 6 | "unanswerable" questions searched against the real index; dropped if the retrieved excerpts answer them |

Lesson: asking the 7B critic for a 1-5 specificity score failed. It rated 130/130 questions "3", even when
its own reason said "specific". Replacing the score with binary checks plus examples fixed this.

Caveat: LLM-generated questions keep lexical overlap with their source passage, which favors BM25.
A human pass over `eval/testset_review.csv` (mark `keep = n`, edit text) is still pending.

## 2. Retrieval (76 answerable questions, no LLM)

Doc-level metrics: the gold article must appear in the top-k results.

| Config | Hit@1 | Hit@5 | Hit@10 | MRR | multi: both docs in top 10 | latency |
|---|---|---|---|---|---|---|
| BM25 | 0.711 | 0.895 | 0.908 | 0.791 | 0.50 | 4 ms |
| Dense (bge-small) | 0.645 | 0.855 | 0.855 | 0.719 | 0.50 | 28 ms |
| Hybrid (RRF) | 0.711 | **0.934** | **0.961** | 0.804 | 0.44 | 19 ms |
| BM25 + rerank | 0.776 | 0.908 | 0.947 | 0.839 | 0.50 | 228 ms |
| Hybrid + rerank (**default**) | 0.776 | 0.908 | 0.947 | 0.839 | 0.50 | 243 ms |
| Hybrid + rerank, 100-word chunks | **0.803** | 0.921 | 0.947 | **0.858** | 0.44 | 179 ms |
| Hybrid + rerank, 350-word chunks | 0.737 | 0.895 | 0.921 | 0.804 | 0.50 | 274 ms |
| Dense (bge-base) | 0.605 | 0.868 | 0.908 | 0.709 | 0.38 | 20 ms |
| Hybrid + rerank (bge-base) | 0.776 | 0.908 | 0.934 | 0.838 | 0.44 | 245 ms |
| Hybrid + rerank + query decomposition | 0.776 | 0.908 | 0.947 | 0.841 | 0.50 | 3.1 s |

Significance testing used a paired bootstrap with 10,000 resamples against BM25
(`retrieval_significance.md`):

- **Hybrid retrieval improves recall: Hit@10 +5.3 pp, 95% CI [+1.3, +10.5].** This is the only
  significant difference.
- The cross-encoder reranker moves the right article to rank 1 more often: Hit@1 +6.6 pp, MRR +4.9 pp. These
  are consistent gains but not significant at n = 76 (the CIs include 0).
- Smaller chunks look best for precision (Hit@1 0.80), but this is also not significant.
- The 3x larger bge-base embedder did not beat bge-small.
- **Weakest point: multi-document questions.** Both gold articles reach the top 10 only ~50% of the time.

## 3. Answers (96 questions; answerer and judge qwen2.5:7b)

| Metric | Baseline (BM25, prompt v1) | Full (hybrid + rerank, v1) | Full + premise check (v2) |
|---|---|---|---|
| Accuracy (correct) | 0.961 | 0.934 | 0.855 |
| Accuracy (partial = 0.5) | 0.974 | 0.967 | 0.928 |
| single / multi accuracy | 0.98 / 0.88 | 0.98 / 0.75 | 0.95 / 0.50 |
| False abstention (answerable) | 0.00 | 0.00 | 0.00 |
| **Correct abstention (unanswerable)** | 0.80 | 0.70 | **0.90** |

### Is the local judge trustworthy?

30 "full" answers were re-judged by `gemini-3.8-flash` (`scripts/validate_judge.py`):

| Judgment | Agreement | Cohen's kappa | Verdict |
|---|---|---|---|
| declined to answer | 0.97 | 0.87 | reliable: the abstention numbers above stand |
| correctness | 0.83 | 0.44 | lenient: 5 answers it called "correct" Gemini rated "partial" |
| any unsupported claim | 0.50 | 0.10 | **unreliable** |

On the same 30 answers:

| | Local judge | Gemini judge |
|---|---|---|
| Accuracy (correct) | 0.90 | **0.77** |
| Faithfulness (claims supported by cited excerpts) | 0.95 | **0.70** |
| Answers with ≥ 1 unsupported claim | 0.07 | **0.57** |

So the local-judge accuracy above is an upper bound, and faithfulness must come from the stronger
judge. Most claims Gemini flags are the 7B answerer's own summarizing statements ("Klarna's
experience highlights..."), but some are real errors, e.g. a made-up "by 2028" target.

Gemini re-judging of the v2 configuration stopped at 5/30 when the free-tier daily quota ran out.
Rerunning `validate_judge.py --config full_premise` after the reset completes it, and cached calls are reused.

## 4. Failure analysis and what changed

1. **False-premise questions.** v1 answered "How did the 2027 export controls affect Nvidia?" by
   restating 2023 facts as if they were about 2027. Prompt v2 adds a premise check: if the question's
   period is later than every excerpt, or the event is never mentioned, it declines. **Correct
   abstention rose from 0.70 to 0.90, with no false abstentions.**
2. **The cost of v2 is multi-document questions.** Multi accuracy fell from 0.75 to 0.50. Inspection shows v2 now
   declines when retrieval found only one of the two needed articles. v1 used to paper over that
   gap. The root cause is retrieval: see section 2, both docs in top 10 ≈ 50%.
3. **Ungrounded synthesis.** The 7B answerer adds interpretive sentences that the excerpts don't state.
4. **Query decomposition: tried, no gain, not shipped.** The LLM splits multi-entity questions into
   ≤ 3 sub-queries, retrieves for each plus the original question, and interleaves the ranked lists
   (`RAGPipeline.retrieve(decompose=True)`). Retrieval was identical to the default:
   - multi "both docs in top 10" stayed at 0.50 (gold docs found per question: 0 → 1, 1 → 7, 2 → 8 in
     both configs);
   - Hit@1 0.776 and MRR 0.841, vs 0.839 for the default;
   - latency rose from 0.24 s to 3.1 s per query.

   Why: of the 7 multi questions missing one article, ~4 retrieved *another article on the same event*
   (e.g. two outlets covering Google pausing Gemini's image generation), so the answer had equivalent
   evidence but the doc-level metric counted a miss. The rest have a second half too vague to point at
   a specific article ("what does this imply about misuse of AI with sensitive data?"). So the multi
   recall gap is partly a labeling artifact, and decomposition doesn't address it. It stays opt-in.

## 5. Next steps

- Event-level gold labels for multi questions: accept any article about the same event, e.g. cluster by
  title embedding + date. This would show the true multi recall.
- Finish Gemini re-judging for v2. Add a stricter "no uncited sentences" rule and measure faithfulness.
- Human review of the test set, plus ~10 hand-written hard questions (`eval/testset_manual.jsonl`).

## Reproduce

```bash
python scripts/build_testset.py --provider ollama --single 120 --multi 40 --unanswerable 40
python scripts/filter_testset.py
python scripts/eval_retrieval.py && python scripts/significance.py
python scripts/eval_answers.py --answer-provider ollama --judge-provider ollama --workers 1
python scripts/validate_judge.py --config full --n 30 --model gemini-3.8-flash
```
