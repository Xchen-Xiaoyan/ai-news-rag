"""Streamlit demo.  Run: streamlit run app/streamlit_app.py"""
import re
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.service import EXAMPLES, get_service  # noqa: E402

st.set_page_config(page_title="AI Industry News Q&A", page_icon="📰", layout="centered",
                   initial_sidebar_state="collapsed")
st.markdown("""<style>
  .block-container {padding-top: 2.5rem; max-width: 760px;}
  .src {padding: .55rem 0; border-bottom: 1px solid rgba(128,128,128,.18);}
  .src-meta {color: #6b7280; font-size: .82rem;}
  .src-snip {color: #4b5563; font-size: .86rem; margin-top: .2rem;}
  .cite {font-size: .78rem; font-weight: 600; padding: 0 .35rem; border-radius: .3rem;
         background: #e8f0fe; color: #1a56db; margin-right: .35rem;}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the news index (about 30 seconds on first start)…")
def service():
    return get_service()


svc = service()
d0, d1 = svc.date_range

# ---------- header ----------
st.title("📰 AI Industry News Q&A")
st.markdown(f"Ask how AI is changing an industry. Answers are written **only from {svc.n_docs:,} news "
            f"articles ({d0:%b %Y} – {d1:%b %Y})**, cite every source, and say so when the archive "
            "doesn't know.")

# ---------- question ----------
if "ask" not in st.session_state:
    st.session_state.ask = ""


def use_example():
    ex = st.session_state.example
    if ex:
        st.session_state.draft = st.session_state.ask = ex
        st.session_state.example = None


with st.form("ask_form", border=False):
    c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
    draft = c1.text_input("Question", key="draft", label_visibility="collapsed",
                          placeholder="e.g. How are banks using generative AI in customer service?")
    if c2.form_submit_button("Ask", type="primary", use_container_width=True):
        st.session_state.ask = draft

st.pills("Try", list(EXAMPLES), format_func=EXAMPLES.get, key="example", on_change=use_example)

with st.expander("Filters"):
    f1, f2 = st.columns(2)
    dates = f1.date_input("Published between", value=(d0, d1), min_value=d0, max_value=d1)
    industries = f2.multiselect("Industry", svc.industries, placeholder="Any",
                                help="Uses the ~3k articles labeled by the original topic-modeling project.")

# ---------- answer ----------
question = st.session_state.ask.strip()
if question:
    date_from, date_to = (str(dates[0]), str(dates[1])) if len(dates) == 2 else (None, None)
    with st.spinner("Searching the archive and writing an answer…"):
        out = svc.ask(question, date_from=date_from, date_to=date_to, industries=industries)

    st.subheader(question)
    with st.container(border=True):
        if out["answer"]:
            if out["abstained"]:
                st.badge("Not in the archive", icon="🔍", color="orange")
            answer = re.sub(r"\[(\d+)\]", lambda m: f"[[{m.group(1)}]](#src-{m.group(1)})", out["answer"])
            st.markdown(answer)
        if out["error"]:
            st.caption(f"⚠️ {out['error']}")
    st.caption(f"Answered in {out['timing']['total_s']:.1f}s"
               + (" · from cache" if out["cached"] else ""))

    # Up to two passages can come from one article: show each article once, with all its numbers.
    articles: dict[str, dict] = {}
    for i, s in enumerate(out["sources"], 1):
        a = articles.setdefault(s["url"], {**s, "nums": []})
        a["nums"].append(i)

    def render(a):
        title = a["title"].split(" | ")[0]
        badges = "".join(f"<span class='cite' id='src-{n}'>{n}</span>" for n in a["nums"])
        st.markdown(f"<div class='src'>{badges}<a href='{a['url']}' target='_blank'>{title}</a>"
                    f"<div class='src-meta'>{a['source']} · {a['date']}</div>"
                    f"<div class='src-snip'>{a['snippet']}</div></div>", unsafe_allow_html=True)

    cited = [a for a in articles.values() if set(a["nums"]) & set(out["cited"])]
    other = [a for a in articles.values() if a not in cited]
    if cited:
        st.markdown(f"**Sources cited** ({len(cited)} articles)")
        for a in cited:
            render(a)
    if other:
        with st.expander(f"{len(other)} more retrieved articles" if cited else f"Retrieved articles ({len(other)})",
                         expanded=not cited):
            for a in other:
                render(a)

# ---------- about ----------
st.divider()
with st.expander("How it works"):
    st.markdown(f"""
1. **Search.** Your question is matched against {svc.n_chunks:,} passages by keywords (BM25) and by meaning
   (bge embeddings). The best 20 are re-scored by a {svc.reranker_name} cross-encoder, and the top 8 are kept.
2. **Answer.** {svc.llm.name} writes an answer using only those passages and cites each claim. If the
   question asks about something the passages don't cover (e.g. a future date), it declines.
3. **Chinese questions** are translated into an English search query first; the answer stays in Chinese.

Evaluated on a 75-question test set. Hit@1 is 0.80 and abstention on unanswerable questions is 100%;
the full report is in `eval/REPORT.md`. LLM calls left today: {svc.budget.limit - svc.budget.used}.
""")
