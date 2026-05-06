"""Streamlit page: body-of-evidence GRADE for a specific outcome.

Wraps grade.py's pipeline: cross-corpus retrieval combined with the per-study
RoB summary (read from outputs/appraisals.csv) feeds an N-run judgment of the
5 downgrade + 3 upgrade GRADE domains, then computes the final certainty.
"""
import csv
import json
import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from grade import (
    _DOWNGRADE_DELTA,
    _UPGRADE_DELTA,
    compute_certainty,
    existing_rob_summary,
    grade_one_domain,
    load_grade_rubric,
)
from grade import write_outputs as write_grade_outputs
from streamlit_clients import get_vector_store

load_dotenv()

st.set_page_config(page_title="GRADE · Burns RAG", page_icon="📈", layout="wide")
st.title("📈 Body-of-Evidence GRADE")
st.caption(
    "Outcome-level certainty assessment across the corpus. "
    "Leans on per-study RoB judgments from the Appraise page (G1 domain)."
)


def _env():
    return {
        "qdrant_path": os.environ.get("QDRANT_PATH", "./qdrant_data"),
        "collection": os.environ.get("COLLECTION_NAME", "burns_papers"),
        "embed_model": os.environ.get("EMBED_MODEL", "text-embedding-3-small"),
        "llm_model": os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        "runs": int(os.environ.get("APPRAISE_RUNS", "10")),
        "temperature": float(os.environ.get("APPRAISE_TEMPERATURE", "0.5")),
        "agreement_threshold": float(os.environ.get("AGREEMENT_THRESHOLD", "0.7")),
        "retrieve_k": int(os.environ.get("GRADE_K", "12")),
    }


def _vector_store():
    return get_vector_store(_env()["collection"])


# --- Sidebar: parameters ---

with st.sidebar:
    st.header("GRADE parameters")
    runs = st.slider("Independent runs per domain", 1, 20, _env()["runs"],
                     help="More runs = tighter agreement estimate, more cost. Spec default = 10.")
    start_from = st.radio(
        "Starting design", ["RCT body", "Non-RCT body"],
        help="GRADE starts at HIGH for RCT bodies, LOW for non-RCT bodies.",
    )
    start_key = "rct" if start_from == "RCT body" else "non_rct"


# --- Inputs ---

outcome = st.text_input(
    "Outcome",
    placeholder="e.g., mortality, infection rate, length of hospital stay",
    help="The clinical outcome whose certainty of evidence you want to grade.",
)

# Hint: warn if no per-study RoB exists yet.
appraisals_csv = Path("outputs") / "appraisals.csv"
if not appraisals_csv.exists():
    st.info(
        "No per-study appraisals found at `outputs/appraisals.csv`. "
        "GRADE will still run, but the G1 (Risk of Bias) domain will lack the "
        "per-study summary it normally leans on. Consider running the **Appraise** "
        "page on a few studies first."
    )

submit = st.button("Run GRADE", type="primary", disabled=not outcome.strip())

st.divider()

if submit:
    vs = _vector_store()
    if vs is None:
        st.error("Qdrant collection not found. Run `py ingest.py` or upload PDFs first.")
        st.stop()

    env = _env()
    env["runs"] = runs
    rubric = load_grade_rubric()
    llm = ChatOpenAI(model=env["llm_model"], temperature=env["temperature"])

    rob_summary = existing_rob_summary(outcome.strip())

    overall_progress = st.progress(0.0)
    status = st.empty()
    results_container = st.container()

    domain_results = []
    for i, d in enumerate(rubric["domains"]):
        status.info(
            f"**[{i+1}/{len(rubric['domains'])}]** {d['id']} · {d['name']} "
            f"({d['kind']}) — running {runs}× …"
        )
        t0 = time.time()
        try:
            res = grade_one_domain(llm, rubric, d, vs, outcome.strip(),
                                   rob_summary, runs=runs, k=env["retrieve_k"])
        except Exception as e:
            st.error(f"Failed on {d['id']}: {e}")
            continue
        res["flagged"] = res["agreement_pct"] < env["agreement_threshold"]
        domain_results.append(res)
        with results_container:
            flag = " 🚩" if res["flagged"] else ""
            st.write(
                f"**{d['id']}** {d['name']} ({d['kind']}) → "
                f"**{res['final_judgment']}** "
                f"({res['agreement_count']}/{runs}){flag} "
                f"in {time.time()-t0:.1f}s"
            )
        overall_progress.progress((i + 1) / len(rubric["domains"]))

    # --- Compute final certainty ---
    start_level = rubric["starting_certainty"][start_key]
    downgrades = [_DOWNGRADE_DELTA.get(d["final_judgment"], 0)
                  for d in domain_results if d["kind"] == "downgrade"]
    upgrades = [_UPGRADE_DELTA.get(d["final_judgment"], 0)
                for d in domain_results if d["kind"] == "upgrade"]
    final = compute_certainty(start_level, downgrades, upgrades)

    parts = [f"Started at {start_level} ({start_from})."]
    for d in domain_results:
        delta = (_DOWNGRADE_DELTA if d["kind"] == "downgrade" else _UPGRADE_DELTA).get(
            d["final_judgment"], 0)
        if delta:
            parts.append(f"{d['name']}: {d['final_judgment']} ({delta:+d}).")
    parts.append(f"Final certainty: {final}.")
    cert_rationale = " ".join(parts)

    grade_result = {
        "outcome": outcome.strip(),
        "started_from": start_level,
        "starting_design": start_key,
        "runs_per_domain": runs,
        "agreement_threshold": env["agreement_threshold"],
        "graded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "domains": domain_results,
        "final_certainty": final,
        "certainty_rationale": cert_rationale,
    }

    status.success(f"Final certainty: **{final}**")

    # --- Final dashboard ---
    st.divider()
    cert_color = {
        "High": "#5cb85c", "Moderate": "#f0ad4e",
        "Low": "#d9534f", "Very low": "#8b0000",
    }.get(final, "#777")
    st.markdown(
        f"""
        <div style="background:{cert_color};padding:1.5rem;border-radius:8px;
                    color:white;text-align:center;">
            <div style="font-size:0.9rem;opacity:0.85;">Outcome: <b>{outcome}</b></div>
            <div style="font-size:2rem;font-weight:bold;margin-top:0.4rem;">
                Certainty of evidence: {final}
            </div>
            <div style="font-size:0.85rem;margin-top:0.5rem;opacity:0.9;">
                {cert_rationale}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Domain table ---
    st.markdown("### Domain breakdown")
    df = pd.DataFrame([
        {
            "Domain": f"{d['domain_id']} · {d['domain_name']}",
            "Kind": d["kind"],
            "Judgment": d["final_judgment"],
            "Agreement": f"{d['agreement_count']}/{runs}",
            "Source papers": ", ".join(d.get("source_documents", []) or [])[:80] +
                             ("…" if len(d.get("source_documents", []) or []) > 4 else ""),
        }
        for d in domain_results
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)

    # --- Per-domain quote drill-down ---
    st.markdown("### Verified evidence (click to expand)")
    for d in domain_results:
        flag = " 🚩" if d["flagged"] else ""
        with st.expander(
            f"{d['domain_id']} · {d['domain_name']} → {d['final_judgment']}"
            f" ({d['agreement_count']}/{runs}){flag}"
        ):
            st.markdown(f"**Rationale:** {d['rationale']}")
            if d["evidence"]:
                st.markdown("**Quotes:**")
                for ev in d["evidence"]:
                    page = ev.get("page", "?")
                    verified = ev.get("verified", False)
                    suffix = "" if verified else " *(unverified)*"
                    st.markdown(f"- _\"{ev['quote']}\"_ — `p.{page}`{suffix}")
            else:
                st.caption("(no quoted evidence; G1 may have used the per-study RoB summary)")

    # --- Persist to disk ---
    try:
        write_grade_outputs(grade_result)
        st.caption(f"Saved JSON + CSV in `outputs/grade_{outcome[:30]}…`")
    except Exception as e:
        st.warning(f"Could not write outputs: {e}")
