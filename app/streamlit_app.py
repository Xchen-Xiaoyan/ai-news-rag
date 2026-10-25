"""Streamlit demo.  Run: streamlit run app/streamlit_app.py"""
import re
import sys
from datetime import date
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.service import EXAMPLES, get_service  # noqa: E402

st.set_page_config(page_title="AI Industry News RAG", page_icon="📰", layout="wide")


@st.cache_resource(show_spinner="Loading 188k-chunk index (first start takes ~30s)…")
def service():
    return get_service()


svc = service()

with st.sidebar:
    st.header("Filters")
    d0, d1 = svc.date_range
    dates = st.date_input("Published between", value=(d0, d1), min_value=d0, max_value=d1)
    industries = st.multiselect("Industry (labeled subset only)", svc.industries,
                                help="Restricts search to the ~3k articles labeled by the original "
                                     "topic-modeling project.")
    k = st.slider("Sources to retrieve", 4, 12, 8)
    st.divider()
    st.caption(f"{svc.n_docs:,} articles · {svc.n_chunks:,} chunks · answers by {svc.llm.name}")
    st.caption(f"LLM calls left today: {svc.budget.limit - svc.budget.used}")
    st.caption(f"Hybrid BM25 + bge-small retrieval, {svc.reranker_name} rerank. See eval/REPORT.md for metrics.")

st.title("📰 AI Industry News — Ask with Citations")
st.write("Ask how AI is affecting an industry. Answers cite the news excerpts they come from, and the "
         "system says so when the archive can't answer.")

if "question" not in st.session_state:
    st.session_state.question = ""

cols = st.columns(3)
for i, ex in enumerate(EXAMPLES):
    if cols[i % 3].button(ex, key=f"ex{i}", use_container_width=True):
        st.session_state.question = ex

question = st.text_input("Your question (English or Chinese)", key="question",
                         placeholder="e.g. How are banks using generative AI in customer service?")

if question:
    date_from, date_to = (str(dates[0]), str(dates[1])) if isinstance(dates, tuple) and len(dates) == 2 else (None, None)
    with st.spinner("Retrieving and answering…"):
        out = svc.ask(question, k=k, date_from=date_from, date_to=date_to, industries=industries)

    if out["answer"]:
        # Turn [3] into a link to the matching source card below.
        answer = re.sub(r"\[(\d+)\]", lambda m: f"[[{m.group(1)}]](#src-{m.group(1)})", out["answer"])
        (st.warning if out["abstained"] else st.success)("Answer" + (" (from cache)" if out["cached"] else ""))
        st.markdown(answer)
    if out["error"]:
        st.info(out["error"])

    t = out["timing"]
    st.caption(f"Search queries: {out['queries']} · retrieval {t['retrieve_s']}s · total {t['total_s']}s")

    st.subheader("Sources")
    for i, s in enumerate(out["sources"], 1):
        cited = "⭐ cited · " if i in out["cited"] else ""
        st.markdown(f"<a name='src-{i}'></a>**[{i}] [{s['title'].split(' | ')[0]}]({s['url']})**  \n"
                    f"<small>{cited}{s['source']} · {s['date']} · rerank score {s['score']:.2f}</small>",
                    unsafe_allow_html=True)
        st.caption(s["snippet"])
