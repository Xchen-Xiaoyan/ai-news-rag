from .clean import split_sentences, is_prose
from . import config


def chunk_text(body: str,
               target: int = config.CHUNK_TARGET_WORDS,
               max_words: int = config.CHUNK_MAX_WORDS,
               overlap: int = config.CHUNK_OVERLAP_SENTENCES,
               min_words: int = config.MIN_CHUNK_WORDS) -> list[str]:
    """Greedy sentence packing into ~target-word chunks with sentence overlap."""
    sents = split_sentences(body)
    chunks, cur, cur_len = [], [], 0
    for s in sents:
        n = len(s.split())
        if cur and cur_len + n > max_words or cur_len >= target:
            chunks.append(cur)
            cur = cur[-overlap:] if overlap else []
            cur_len = sum(len(x.split()) for x in cur)
        cur.append(s)
        cur_len += n
    if cur and (not chunks or cur_len > sum(len(x.split()) for x in chunks[-1][-overlap:])):
        chunks.append(cur)

    out = []
    for c in chunks:
        # Drop chunks that are mostly leftover boilerplate (footers, related-article lists).
        if sum(map(is_prose, c)) / len(c) < 0.5:
            continue
        text = " ".join(c)
        if len(text.split()) >= min_words:
            out.append(text)
    return out
