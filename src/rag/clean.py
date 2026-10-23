"""Sentence-level boilerplate removal for scraped news text.

The raw text has no newlines: navigation menus, share buttons and footers are
glued to the article body. We split into sentence-like segments, score each one
for "prose-likeness", and keep the longest contiguous run of prose.
"""
import re

STOPWORDS = set(
    "the a an of to and in is are was were be been for on that this with as by at from it its "
    "or but not have has had will would can could their they we you he she his her our which "
    "who what when how than more about into also after over said says".split()
)

_SENT_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")
_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def normalize(text: str) -> str:
    # Paragraph breaks were lost in scraping, gluing sentences: "profits.Chief" -> "profits. Chief".
    # Requiring a lowercase/digit before the period keeps "U.S." and "bloomberg.com" intact.
    text = re.sub(r"([a-z0-9%)\]”’\"])([.!?])([“\"‘(]?[A-Z])", r"\1\2 \3", text)
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def is_prose(sentence: str) -> bool:
    words = _WORD.findall(sentence)
    n = len(words)
    if n < 5 or n > 120:
        return False
    stop_ratio = sum(w.lower() in STOPWORDS for w in words) / n
    cap_ratio = sum(w[0].isupper() for w in words) / n
    return stop_ratio >= 0.15 and cap_ratio <= 0.5 and sentence[-1] in ".!?\"'”)"


def extract_body(text: str, title: str = "", min_run: int = 3) -> str:
    """Return the article body: from the first to the last run of >= min_run prose sentences."""
    sents = split_sentences(normalize(text))
    flags = [is_prose(s) for s in sents]

    # Find runs of consecutive prose sentences; tolerate one non-prose sentence inside a run.
    runs, start, gap = [], None, 0
    for i, ok in enumerate(flags + [False, False]):
        if ok:
            if start is None:
                start = i
            gap = 0
            end = i
        elif start is not None:
            gap += 1
            if gap > 1:
                runs.append((start, end))
                start, gap = None, 0
    runs = [r for r in runs if sum(flags[r[0]:r[1] + 1]) >= min_run]
    if not runs:
        return ""
    body = " ".join(sents[runs[0][0]: runs[-1][1] + 1])

    # Drop a leading copy of the title if the body starts with it.
    t = normalize(title).split(" | ")[0].strip()
    if t and body.startswith(t):
        body = body[len(t):].lstrip(" .:-")
    return body
