"""Streamlit page: grounded Q&A over the corpus or appraisal index.

Wraps query.py's verification pipeline (structured output + quote-substring
match + number-supported check) and renders the results as a UI.
"""
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from query import (
    GroundedAnswer,
    SYSTEM_PROMPTS,
    build_chain,
    get_env,
)
from streamlit_clients import get_vector_store

load_dotenv()

st.set_page_config(page_title="Query · Burns RAG", page_icon="🔍", layout="wide")
st.title("🔍 Grounded Q&A")
st.caption(
    "Every answer is backed by verbatim quotes from the source PDFs. "
    "Claims that can't be verified are refused, not paraphrased."
)


# --- Sidebar controls ---

with st.sidebar:
    st.header("Settings")
    collection_label = st.radio(
        "Knowledge base",
        ["Papers (PDF chunks)", "Appraisals (rubric judgments)"],
        help=(
            "Switch between the original PDF corpus and the index of structured "
            "rubric judgments produced by the Appraise page."
        ),
    )
    collection = "papers" if collection_label.startswith("Papers") else "appraisals"

    st.divider()
    env_preview = get_env()
    st.caption(
        f"**Models**\n\n"
        f"- Answer: `{env_preview['answer_model']}`\n"
        f"- Reranker: `{env_preview['rerank_model']}`\n\n"
        f"Override via `QUERY_MODEL` and `LLM_MODEL` in `.env`."
    )

    st.divider()
    st.markdown(
        "**Verification rules**\n\n"
        "- Every numeric claim must appear verbatim in a cited quote.\n"
        "- Quotes that don't substring-match the retrieved chunks are dropped.\n"
        "- If >50% of citations fail, the entire answer is refused."
    )


# --- Build chain (cached so we don't re-instantiate on every keystroke) ---

@st.cache_resource(show_spinner="Loading retriever…")
def _build(coll: str):
    env = get_env()
    target = env["papers_collection"] if coll == "papers" else env["appraisals_collection"]
    vs = get_vector_store(target)
    if vs is None:
        hint = "`py ingest.py`" if coll == "papers" else "`py appraise.py --reindex`"
        raise SystemExit(f"Collection '{target}' not found. Run {hint} first.")
    return build_chain(env, coll, vector_store=vs)


try:
    run = _build(collection)
except SystemExit as e:
    st.error(str(e))
    st.stop()


# --- Question input ---

question = st.text_area(
    "Question",
    placeholder=(
        "e.g., What proportion of burn injuries occur in men versus women across studies?"
    ),
    height=80,
)

submit = st.button("Ask", type="primary", disabled=not question.strip())

st.divider()

if submit:
    with st.spinner("Retrieving and grounding the answer…"):
        try:
            result, verified, unverified, unsupported, _docs = run(question.strip())
        except Exception as e:
            st.exception(e)
            st.stop()

    # --- Render result ---

    if result.refused:
        st.warning("**Refused — corpus does not contain a clear answer.**")
        if result.refusal_reason:
            st.caption(f"Reason: {result.refusal_reason}")
        st.stop()

    total = len(result.citations)
    if total == 0:
        st.error(
            "**Refused — model produced an answer with no citations.** "
            "This is treated as ungrounded."
        )
        with st.expander("Draft (rejected)"):
            st.write(result.answer)
        st.stop()

    too_short = [u for u in unverified if u.get("reason") == "too_short"]
    not_found = [u for u in unverified if u.get("reason") == "not_found"]

    if len(verified) == 0:
        if too_short and not not_found:
            st.warning(
                f"**Refused — model returned only fragmentary quotes ({len(too_short)} too short to verify).** "
                "Quotes like *\"69.1 days\"* are too short to uniquely identify a passage and "
                "would substring-match unrelated chunks (tables, ages, etc.). The numbers may be "
                "real, but the model didn't quote enough context. Try rephrasing the question "
                "or asking for a more specific outcome."
            )
        else:
            st.error(
                "**Refused — the model cited sources, but none of the quotes appear "
                "verbatim in the retrieved chunks.** This is the typical fingerprint "
                "of fabrication."
            )
        with st.expander(f"Draft (rejected) — {total} unverified citations"):
            st.write(result.answer)
            if too_short:
                st.markdown("**Too short to verify (need 10+ words of context):**")
                for u in too_short:
                    st.markdown(f"- _\"{u['quote']}\"_")
            if not_found:
                st.markdown("**Not found in retrieved chunks (possible fabrication):**")
                for u in not_found:
                    st.markdown(f"- _\"{u['quote']}\"_")
        st.stop()

    if len(unverified) / total > 0.5:
        st.error(
            f"**Refused** — {len(unverified)}/{total} citations failed verification "
            f"({len(too_short)} too short, {len(not_found)} not found). "
            "Treating the answer as ungrounded."
        )
        with st.expander("Draft (rejected)"):
            st.write(result.answer)
        st.stop()

    # Acceptable answer.
    st.subheader("Answer")
    st.markdown(result.answer)

    if unsupported:
        st.warning(
            f"⚠️ {len(unsupported)} number(s) in the answer have no matching quote "
            f"in the verified evidence: **{', '.join(unsupported)}**.\n\n"
            "Re-check against the source PDF before citing."
        )

    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.subheader(f"Verified evidence ({len(verified)}/{total})")
        for i, v in enumerate(verified, 1):
            st.markdown(
                f"**{i}.** _\"{v['quote']}\"_  \n"
                f"&nbsp;&nbsp;&nbsp;&nbsp;`{v['source']} — page {v['page']}`"
            )

    with col_r:
        if unverified:
            st.subheader(f"Dropped ({len(unverified)})")
            for u in unverified:
                q = u["quote"]
                tag = "too short" if u.get("reason") == "too_short" else "not in chunks"
                st.markdown(f"- _\"{q[:160]}{'…' if len(q) > 160 else ''}\"_  \n"
                            f"&nbsp;&nbsp;&nbsp;&nbsp;`[{tag}]`")
        else:
            st.subheader("Dropped")
            st.caption("None — every cited quote was verified.")
