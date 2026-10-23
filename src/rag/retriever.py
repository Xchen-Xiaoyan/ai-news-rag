"""Hybrid retrieval: BM25 + dense, fused with RRF, optionally reranked by a cross-encoder."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from . import config
from .index import bm25_query_tokens, load_bm25, load_dense, passage_text


@dataclass
class Filters:
    date_from: str | None = None   # "YYYY-MM-DD"
    date_to: str | None = None
    industries: list[str] = field(default_factory=list)  # match any (high_conf labels)
    sentiment: str | None = None
    high_conf_only: bool = False


def rrf(rankings: list[np.ndarray], k: int = 60) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[int(idx)] = scores.get(int(idx), 0.0) + 1.0 / (k + rank + 1)
    return scores


class Retriever:
    def __init__(self, embed_model: str = config.EMBED_MODEL,
                 rerank_model: str | None = config.RERANK_MODEL, device: str | None = None):
        from sentence_transformers import CrossEncoder, SentenceTransformer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.docs = pd.read_parquet(config.DOCS).set_index("doc_id")
        self.chunks = pd.read_parquet(config.CHUNKS)
        self.passages = passage_text(self.chunks, self.docs.reset_index())
        self.chunk_doc = self.chunks.doc_id.to_numpy()

        self.bm25 = load_bm25(config.INDEX)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.dense = torch.from_numpy(load_dense(config.INDEX, embed_model)).to(self.device, dtype)
        self.encoder = SentenceTransformer(embed_model, device=self.device)
        self.reranker = CrossEncoder(rerank_model, device=self.device, max_length=512) if rerank_model else None
        if self.device == "cuda":
            self.encoder.half()
            if self.reranker:
                self.reranker.model.half()

    # ---------- candidate generation ----------
    def _mask(self, f: Filters | None) -> np.ndarray | None:
        if f is None:
            return None
        d = self.docs
        keep = pd.Series(True, index=d.index)
        if f.date_from:
            keep &= d.date >= pd.Timestamp(f.date_from)
        if f.date_to:
            keep &= d.date <= pd.Timestamp(f.date_to)
        if f.high_conf_only or f.industries or f.sentiment:
            keep &= d.in_high_conf
        if f.industries:
            wanted = set(f.industries)
            keep &= d.industries.map(lambda xs: xs is not None and bool(wanted & set(xs)))
        if f.sentiment:
            keep &= d.sentiment == f.sentiment
        if keep.all():
            return None
        return keep.reindex(self.chunk_doc).to_numpy()

    def bm25_search(self, query: str, n: int, mask: np.ndarray | None = None) -> np.ndarray:
        tokens = bm25_query_tokens(query)
        if not tokens:
            return np.array([], dtype=int)
        scores = self.bm25.get_scores(tokens)
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        top = np.argpartition(-scores, min(n, len(scores) - 1))[:n]
        top = top[np.argsort(-scores[top])]
        return top[scores[top] > 0]

    def dense_search(self, query: str, n: int, mask: np.ndarray | None = None) -> np.ndarray:
        q = self.encoder.encode([config.QUERY_INSTRUCTION + query], normalize_embeddings=True,
                                convert_to_tensor=True).to(self.dense.dtype)
        scores = (self.dense @ q.T).squeeze(1).float()
        if mask is not None:
            scores[~torch.from_numpy(mask).to(scores.device)] = -torch.inf
        top = torch.topk(scores, n).indices.cpu().numpy()
        return top[np.isfinite(scores[top].cpu().numpy())]

    # ---------- public API ----------
    def search(self, query: str, k: int = 8, mode: str = "hybrid", rerank: bool = True,
               filters: Filters | None = None, n_candidates: int = 50,
               max_per_doc: int = 2) -> list[dict]:
        mask = self._mask(filters)
        if mode == "bm25":
            cands = list(self.bm25_search(query, n_candidates, mask))
        elif mode == "dense":
            cands = list(self.dense_search(query, n_candidates, mask))
        elif mode == "hybrid":
            fused = rrf([self.bm25_search(query, 100, mask), self.dense_search(query, 100, mask)])
            cands = sorted(fused, key=fused.get, reverse=True)[:n_candidates]
        else:
            raise ValueError(f"unknown mode {mode}")

        scores = {i: 1.0 / (r + 1) for r, i in enumerate(cands)}
        if rerank and self.reranker and cands:
            rs = self.reranker.predict([(query, self.passages[i]) for i in cands], batch_size=32,
                                       show_progress_bar=False)
            scores = dict(zip(cands, map(float, rs)))
            cands = sorted(cands, key=scores.get, reverse=True)

        # Keep at most max_per_doc chunks per article so answers draw on several sources.
        out, per_doc = [], {}
        for i in cands:
            doc_id = self.chunk_doc[i]
            if per_doc.get(doc_id, 0) >= max_per_doc:
                continue
            per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
            out.append(self._result(i, scores[i]))
            if len(out) == k:
                break
        return out

    def _result(self, i: int, score: float) -> dict:
        c = self.chunks.iloc[i]
        d = self.docs.loc[c.doc_id]
        return {
            "chunk_id": c.chunk_id, "doc_id": c.doc_id, "score": round(score, 4),
            "title": d.title, "url": d.url, "source": d.source,
            "date": d.date.strftime("%Y-%m-%d"), "text": c.text,
        }
