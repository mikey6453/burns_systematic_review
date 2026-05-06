"""Streamlit frontend entry point. Run with:  streamlit run app.py

The sidebar nav is auto-populated from files in pages/.
"""
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Burns RAG — Evidence Appraisal",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Burns RAG — Evidence Appraisal")
st.caption("Retrieval-augmented Q&A and rubric-based grading for systematic-review PDFs.")

# Quick sanity panel up top so the user knows the system is wired up.
col1, col2, col3 = st.columns(3)
with col1:
    api_set = bool(os.environ.get("OPENAI_API_KEY", "").startswith("sk-"))
    st.metric("OpenAI API key", "Configured" if api_set else "Missing")
with col2:
    qdrant_path = os.environ.get("QDRANT_PATH", "./qdrant_data")
    has_index = (Path(qdrant_path) / "collection").exists()
    st.metric("Qdrant index", "Ready" if has_index else "Not built")
with col3:
    kb = Path(os.environ.get("KNOWLEDGE_BASE_PATH", "./kb"))
    pdf_count = len(list(kb.glob("*.pdf"))) if kb.exists() else 0
    st.metric("Papers in kb/", pdf_count)

st.divider()

st.markdown(
    """
### What this tool does

**1. Grounded Q&A (Query page)**
Ask natural-language questions over the entire corpus. Answers come with
verbatim quote citations — every factual claim is verified against the
source PDF before being shown. Hallucinated content is automatically
refused.

**2. Rubric-based appraisal (Appraise page)**
Upload a folder of studies, pick a rubric (RoB 2 / ROBINS-I /
Newcastle-Ottawa / custom), and let the AI grade each paper across
multiple independent runs. The system reports inter-run agreement,
flags disagreements for human review, and produces a publication-style
traffic-light plot.

**3. Body-of-evidence GRADE (GRADE page)**
For a specific outcome (e.g., mortality, infection rate), assess the
overall certainty of evidence across the corpus using the GRADE
framework. Outputs HIGH / MODERATE / LOW / VERY LOW with a
domain-by-domain rationale.

---

### How to use

Use the sidebar to navigate between pages. Suggested order if you're
starting fresh:

1. Run `py ingest.py` from a terminal once to build the local Qdrant index
   from the PDFs in `kb/`.
2. Open the **Query** page to verify the index is healthy.
3. Open the **Appraise** page to grade studies. Upload more PDFs from the
   UI if you want to extend the corpus on the fly.
4. Open the **GRADE** page once you have at least a few appraisals to
   summarize body-of-evidence certainty by outcome.
"""
)

st.divider()
st.caption(
    f"Working directory: `{Path.cwd()}` · "
    f"LLM: `{os.environ.get('LLM_MODEL', 'gpt-4o-mini')}` · "
    f"Embeddings: `{os.environ.get('EMBED_MODEL', 'text-embedding-3-small')}`"
)
