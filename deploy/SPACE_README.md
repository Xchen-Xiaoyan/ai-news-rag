---
title: AI Industry News RAG
emoji: 📰
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Cited answers about AI's impact on industries (33k articles)
---

Cited answers about how AI is changing industries, from 33,234 news articles (2022–2026).
Hybrid BM25 + dense retrieval with a cross-encoder reranker, and a Gemini answerer that declines
false-premise questions.

Code, evaluation report and design notes: https://github.com/Xchen-Xiaoyan/ai-news-rag

Space secrets: `GEMINI_API_KEY`, `HF_TOKEN` (read access to the private data repo).
Variables: `HF_DATA_REPO`, `RAG_DEVICE=cpu`, `MAX_DAILY_LLM_CALLS`.
