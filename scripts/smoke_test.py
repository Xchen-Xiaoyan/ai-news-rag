import sys, time; sys.path.insert(0, ".")
from dotenv import load_dotenv; load_dotenv()
from src.rag.retriever import Retriever
from src.rag.llm import get_llm
from src.rag.pipeline import RAGPipeline
rag = RAGPipeline(Retriever(), get_llm())
for q in [
          "AI 对客服和呼叫中心岗位有什么影响?",
          "What was Apple's quarterly revenue from its Vision Pro headset in Q3 2025?"]:
    t = time.time(); o = rag.answer(q)
    print(f"\n### {q}  ({time.time()-t:.1f}s, cited={o['cited']}, abstained={o["abstained"]}, queries={o["queries"]!r})\n{o['answer']}")
    for i in o["cited"][:4]:
        h = o["sources"][i-1]; print(f"   [{i}] {h['title'][:80]} ({h['source']}, {h['date']})")
