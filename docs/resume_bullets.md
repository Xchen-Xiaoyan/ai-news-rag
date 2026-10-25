# 简历条目 / Resume bullets

按岗位选 3–4 条。数字都能在 `eval/REPORT.md` 里找到出处，面试时可以逐条讲清楚。

## English

**AI Industry News RAG: cited Q&A over 33k news articles** · Python, PyTorch, sentence-transformers, BM25, FastAPI, Streamlit, Ollama, Gemini

- Built a retrieval-augmented QA system on a 135k-article AI-news corpus. Wrote sentence-level boilerplate
  removal and syndication dedup to reach 33k articles and 188k chunks, then built a hybrid BM25 + dense retriever with a
  cross-encoder reranker (Hit@1 0.80, Hit@10 0.98).
- Designed an evaluation harness covering single-doc, multi-doc and unanswerable questions. LLM-generated
  candidates passed through rule, LLM-critic and retrieval-probe filters (207 → 75 questions). Ablations covered
  retriever, chunk size, embedder and reranker, with paired-bootstrap confidence intervals.
- Found that dense-only retrieval was significantly worse than BM25 on news (Hit@10 −8.5 pp, 95% CI
  [−15.3, −1.7]). The reranker improved Hit@1 by 7–10 pp, and query decomposition gave no gain, so it was
  not shipped.
- Diagnosed false-premise hallucinations ("How did the 2027 export controls…") and added a premise check
  to the prompt. Correct abstention rose from 81% to 100% with zero false abstentions.
- Validated the local 7B LLM judge against a stronger model. It was reliable for abstention (κ 0.87) but
  missed most unsupported claims (κ 0.10; 7% vs 57% hallucination rate), so faithfulness is reported from the
  stronger judge only.
- Shipped a $0 demo with FastAPI + Streamlit: cached LLM calls, a per-model free-tier fallback chain, a daily
  budget, and retrieval-only degradation. Cut CPU latency from ~11 s to 2.8 s by swapping the reranker
  after measuring the accuracy trade-off.

## 中文

**AI 行业新闻 RAG 问答系统（带引用）**｜Python、PyTorch、BM25、sentence-transformers、FastAPI、Streamlit

- 基于 13.5 万篇 AI 新闻语料构建 RAG 问答系统：句子级网页去噪 + 转载去重 → 3.3 万篇 / 18.8 万 chunk；BM25 + 向量混合检索 + 交叉编码器重排，Hit@1 0.80、Hit@10 0.98
- 设计评测体系：单文档 / 跨文档 / 无法回答三类问题，LLM 生成候选题后经规则、LLM 评审、检索反证三道质检（207 → 75 题）；对检索方式、chunk 大小、向量模型、重排模型做消融，并用配对 bootstrap 给出置信区间
- 发现纯向量检索在新闻语料上显著差于 BM25（Hit@10 −8.5pp，95% CI [−15.3, −1.7]）；重排序使 Hit@1 提升 7–10pp；问题分解无增益，因而未上线
- 定位"错误前提"类幻觉（如"2027 年出口管制如何…"），加入前提校验后正确拒答率 81% → 100%，误拒率为 0
- 用更强模型校验本地 7B 裁判：拒答判断可靠（κ 0.87），幻觉判断严重偏松（κ 0.10；7% vs 57%），因此只采用强裁判的忠实度结果
- 零成本上线 demo：LLM 调用缓存、按模型的免费额度降级链、每日调用上限、LLM 不可用时退化为只展示检索结果；量化精度代价后更换重排模型，CPU 端到端延迟从约 11 秒降到 2.8 秒

## 面试时可以讲的故事（STAR）

1. **"不要相信自己的指标"**：本地裁判给出 95% 准确率、4% 幻觉率，我没有直接采用，而是抽样请更强的模型复核，发现幻觉被严重低估（κ 0.10），于是改变了报告方式。
2. **"改进有代价，而且要量化"**：前提校验让拒答率达到 100%，但跨文档题准确率下降了。我追查原因，发现问题出在检索没找全；再进一步发现，其中很大一部分是测试集标注的问题，不是模型的问题。
3. **"负面结果也是结果"**：问题分解没有带来任何提升，延迟还增加了一倍。我没有上线它，而是把原因写进了报告。
4. **"工程约束下的取舍"**：免费 CPU 环境下，重排模型是唯一的瓶颈。我比较了截断长度、减少候选、换小模型三种方案的精度和延迟，最后选择了换小模型。
