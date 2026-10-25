"""Retrieve -> prompt with numbered sources -> cited answer."""
import re

from .llm import LLM
from .retriever import Filters, Retriever

SYSTEM_V1 = """You are an analyst answering questions about how AI is affecting industries, using ONLY \
the numbered news excerpts provided.

Rules:
- Every factual sentence must end with citations like [1] or [2][4] pointing to the excerpts that support it.
- Do not use outside knowledge. If the excerpts do not contain enough information, say \
"The corpus does not contain enough information to answer this." and stop.
- Prefer concrete details from the excerpts: companies, numbers, dates, mechanisms.
- If sources disagree, say so and cite both sides.
- Be concise: at most ~200 words, in the same language as the question."""

# v2 adds a premise check. Eval found v1 answering false-premise questions ("How did the 2027 export
# controls...?") by projecting older facts onto the asked-about date.
SYSTEM = SYSTEM_V1 + """
- Check the question's premise first. If it asks about a time period later than every excerpt's date, or about an event, figure or relationship the excerpts never mention, the excerpts cannot answer it: reply "The corpus does not contain enough information to answer this." and then, in at most one sentence, mention the closest related information the excerpts do contain, with citations."""

NO_ANSWER = "The corpus does not contain enough information"

REWRITE_SYSTEM = """Translate the user's question into a concise English search query for a news \
archive about AI and industry. Output only the query, nothing else."""

_CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")


def format_sources(hits: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] {h['title'].split(' | ')[0]} ({h['source']}, {h['date']})\n{h['text']}"
        for i, h in enumerate(hits, 1))


def parse_citations(answer: str, n_sources: int) -> list[int]:
    cited = {int(x) for x in re.findall(r"\[(\d+)\]", answer)}
    return sorted(c for c in cited if 1 <= c <= n_sources)


class RAGPipeline:
    def __init__(self, retriever: Retriever, llm: LLM, system: str = SYSTEM):
        self.retriever, self.llm, self.system = retriever, llm, system

    def search_query(self, question: str) -> str:
        # The corpus and both indexes are English-only, so non-English questions are translated first.
        if _CJK.search(question):
            return self.llm.generate(REWRITE_SYSTEM, question).strip()
        return question

    def answer(self, question: str, k: int = 8, filters: Filters | None = None, **search_kw) -> dict:
        query = self.search_query(question)
        hits = self.retriever.search(query, k=k, filters=filters, **search_kw)
        return {"query": query, **self.generate(question, hits)}

    def generate(self, question: str, hits: list[dict]) -> dict:
        """Answer from already-retrieved hits (lets evals retrieve serially and generate in parallel)."""
        if not hits:
            return {"question": question, "answer": NO_ANSWER + " to answer this.",
                    "sources": [], "cited": [], "abstained": True, "llm": self.llm.name}
        prompt = f"Excerpts:\n\n{format_sources(hits)}\n\nQuestion: {question}"
        answer = self.llm.generate(self.system, prompt).strip()
        return {
            "question": question, "answer": answer, "sources": hits,
            "cited": parse_citations(answer, len(hits)),
            "abstained": NO_ANSWER.lower() in answer.lower(),
            "llm": self.llm.name,
        }
