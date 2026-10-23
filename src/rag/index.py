"""Build and load the sparse (BM25) and dense (bge) indexes over chunks."""
from pathlib import Path

import bm25s
import numpy as np
import pandas as pd
import Stemmer

from . import config

_stemmer = Stemmer.Stemmer("english")


def passage_text(chunks: pd.DataFrame, docs: pd.DataFrame) -> list[str]:
    """Prepend the article title so chunks keep their context ("the company" -> which company)."""
    titles = chunks.doc_id.map(docs.set_index("doc_id").title.str.split(" | ", regex=False).str[0])
    return (titles + "\n" + chunks.text).tolist()


def bm25_query_tokens(query: str) -> list[str]:
    return bm25s.tokenize([query], stopwords="en", stemmer=_stemmer,
                          return_ids=False, show_progress=False)[0]


def build_bm25(texts: list[str], out_dir: Path):
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(texts, stopwords="en", stemmer=_stemmer))
    retriever.save(str(out_dir / "bm25"))


def load_bm25(index_dir: Path) -> bm25s.BM25:
    return bm25s.BM25.load(str(index_dir / "bm25"))


def build_dense(texts: list[str], model_name: str, out_dir: Path, batch_size: int = 256):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device="cuda")
    model.half()
    emb = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                       show_progress_bar=True, convert_to_numpy=True)
    np.save(out_dir / f"dense_{model_name.split('/')[-1]}.npy", emb.astype(np.float16))


def load_dense(index_dir: Path, model_name: str) -> np.ndarray:
    return np.load(index_dir / f"dense_{model_name.split('/')[-1]}.npy")
