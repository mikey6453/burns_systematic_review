"""Streamlit page: full evidence-appraisal workflow.

A 4-step wizard that mirrors the spec:

    1. Pick papers      — choose from the existing index OR upload new PDFs
    2. Pick rubrics     — auto-detect study design, override per paper, or
                          apply one rubric to all. Custom rubrics can be
                          uploaded as JSON.
    3. Run grading      — N independent runs per domain at temperature>0,
                          with live progress and inter-run agreement
    4. Review & adjudicate — traffic-light plot + per-paper drill-down;
                             flagged (low-agreement) judgments get a
                             tie-breaker UI for the user to set the final
                             rating.
"""
import csv
import json
import os
import shutil
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from appraise import (
    APPRAISALS_CSV,
    OUTPUTS_DIR,
    RUBRICS_DIR,
    detect_design,
    grade_paper,
    list_papers,
    load_rubric,
    rebuild_summary_csv,
    reindex_appraisals,
    write_outputs,
)
from ingest import ingest_files
from streamlit_clients import get_qdrant_client, get_vector_store
from visualize import plot as visualize_plot

load_dotenv()

st.set_page_config(page_title="Appraise · Burns RAG", page_icon="📋", layout="wide")
st.title("📋 Evidence Appraisal")
st.caption(
    "Multi-run rubric grading with quote-verified evidence. "
    "Judgments below the agreement threshold are flagged for human adjudication."
)


# --- Env + clients (cached) ---

def _env():
    return {
        "qdrant_path": os.environ.get("QDRANT_PATH", "./qdrant_data"),
        "collection": os.environ.get("COLLECTION_NAME", "burns_papers"),
        "appraisals_collection": os.environ.get("APPRAISALS_COLLECTION", "burns_appraisals"),
        "embed_model": os.environ.get("EMBED_MODEL", "text-embedding-3-small"),
        "llm_model": os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        "detect_model": os.environ.get("DETECT_MODEL", "gpt-4o"),
        "runs": int(os.environ.get("APPRAISE_RUNS", "10")),
        "temperature": float(os.environ.get("APPRAISE_TEMPERATURE", "0.5")),
        "agreement_threshold": float(os.environ.get("AGREEMENT_THRESHOLD", "0.7")),
        "retrieve_k": int(os.environ.get("APPRAISE_K", "6")),
    }


def _vector_store():
    """Return (vector_store_or_None, client) using the shared singleton client."""
    env = _env()
    client = get_qdrant_client()
    vs = get_vector_store(env["collection"])
    return vs, client


def _list_rubrics() -> list[str]:
    """Study-level rubrics only (GRADE lives on its own page)."""
    rubrics = []
    for p in sorted(RUBRICS_DIR.glob("*.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if r.get("level", "study") == "study":
            rubrics.append(p.stem)
    return rubrics


def _kb_dir() -> Path:
    p = Path(os.environ.get("KNOWLEDGE_BASE_PATH", "./kb"))
    p.mkdir(exist_ok=True)
    return p


# --- Session state ---

S = st.session_state
S.setdefault("step", "pick")
S.setdefault("selected", [])
S.setdefault("paper_rubrics", {})       # {filename: rubric_id}
S.setdefault("detection_results", {})   # {filename: {confidence, rationale}}
S.setdefault("appraisals", {})          # {filename: appraisal_dict}
S.setdefault("adjudications", {})       # {(filename, domain_id): final_judgment}


# --- Stepper ---

steps = ["pick", "rubric", "grade", "review"]
step_labels = ["1. Pick papers", "2. Pick rubrics", "3. Run grading", "4. Review & adjudicate"]
cols = st.columns(len(steps))
for i, (k, label) in enumerate(zip(steps, step_labels)):
    with cols[i]:
        is_current = S.step == k
        is_done = steps.index(S.step) > i
        marker = "🔷" if is_current else ("✅" if is_done else "⚪")
        st.markdown(f"### {marker} {label}")

st.divider()


# ============================================================
# STEP 1 — Pick papers
# ============================================================

if S.step == "pick":
    st.subheader("Pick papers to grade")

    vs, client = _vector_store()
    if vs is None:
        st.error(
            "No Qdrant collection found. Run `py ingest.py` first, or upload "
            "papers below."
        )
    available = list_papers(client, _env()["collection"]) if vs is not None else []

    tab_existing, tab_upload = st.tabs(
        [f"Existing corpus ({len(available)} papers)", "Upload new PDFs"]
    )

    with tab_existing:
        if not available:
            st.info("Nothing in the index yet. Use the **Upload** tab.")
        else:
            picked = st.multiselect(
                "Select papers to grade",
                options=available,
                default=S.selected,
                help="Multi-select. Use the search box to filter.",
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Select all"):
                    picked = available
            with c2:
                if st.button("Clear"):
                    picked = []
            S.selected = picked
            st.caption(f"{len(S.selected)} paper(s) selected")

    with tab_upload:
        uploaded = st.file_uploader(
            "Drop PDF files here",
            type=["pdf"],
            accept_multiple_files=True,
            help=(
                "Files are saved to `kb/`, then chunked and embedded into the "
                "existing Qdrant collection. Duplicates (by filename) are skipped."
            ),
        )
        if uploaded and st.button("Ingest these PDFs", type="primary"):
            kb = _kb_dir()
            saved_paths: list[Path] = []
            for f in uploaded:
                target = kb / f.name
                target.write_bytes(f.getbuffer())
                saved_paths.append(target)

            progress = st.progress(0.0)
            status = st.empty()

            def cb(stage, payload):
                if stage == "file_start":
                    status.info(
                        f"Embedding **{payload['path'].name}** "
                        f"({payload['index']+1}/{payload['total']})…"
                    )
                elif stage == "file_done":
                    progress.progress((payload["index"] + 1) / payload["total"]
                                      if payload["total"] else 1.0)
                elif stage == "file_skip":
                    status.warning(f"Skipped {payload['path'].name}: {payload['reason']}")

            with st.spinner("Ingesting…"):
                summary = ingest_files(saved_paths, _env(), progress_cb=cb)

            progress.progress(1.0)
            status.success(
                f"Done. Ingested {len(summary['ingested'])} new papers "
                f"({summary['chunks']} chunks). Skipped {len(summary['skipped'])}."
            )
            # Add freshly-ingested papers to the selection by default.
            for fn in summary["ingested"]:
                if fn not in S.selected:
                    S.selected.append(fn)
            # Drop the vector_store cache so list_papers picks up the new docs.
            get_vector_store.clear()
            time.sleep(0.5)
            st.rerun()

    st.divider()
    if st.button("Continue →", type="primary", disabled=not S.selected):
        S.step = "rubric"
        st.rerun()


# ============================================================
# STEP 2 — Pick rubrics
# ============================================================

elif S.step == "rubric":
    st.subheader("Pick a rubric for each paper")
    rubrics = _list_rubrics()

    st.markdown(
        "The tool can recommend a rubric based on each paper's study design "
        "(RCT → RoB 2, non-randomized intervention → ROBINS-I, observational "
        "cohort → Newcastle-Ottawa). You can override any recommendation, or "
        "apply one rubric to all selected papers below."
    )

    c_left, c_right = st.columns([2, 1])
    with c_left:
        bulk_rubric = st.selectbox(
            "Apply rubric to ALL selected papers",
            options=["(don't override)"] + rubrics,
            index=0,
        )
        if st.button("Apply", disabled=bulk_rubric == "(don't override)"):
            for fn in S.selected:
                S.paper_rubrics[fn] = bulk_rubric
            st.rerun()

    with c_right:
        custom = st.file_uploader(
            "Upload a custom rubric (JSON)",
            type=["json"],
            help=(
                "Drop in a JSON file matching the schema in rubrics/custom.json. "
                "It will be saved to rubrics/ and become available for use."
            ),
        )
        if custom is not None:
            try:
                payload = json.loads(custom.getvalue().decode("utf-8"))
                rid = payload.get("id")
                if not rid:
                    st.error("Rubric JSON must include an 'id' field.")
                else:
                    (RUBRICS_DIR / f"{rid}.json").write_bytes(custom.getbuffer())
                    st.success(f"Saved custom rubric: {rid}")
                    st.rerun()
            except Exception as e:
                st.error(f"Invalid rubric JSON: {e}")

    st.divider()

    # Auto-detect button + per-paper override grid.
    env = _env()
    if st.button(f"Auto-detect rubric for all papers (using {env['detect_model']})"):
        vs, _ = _vector_store()
        llm = ChatOpenAI(model=env["detect_model"], temperature=0)
        progress = st.progress(0.0)
        status = st.empty()
        for i, fn in enumerate(S.selected):
            status.info(f"Classifying **{fn}** ({i+1}/{len(S.selected)})…")
            try:
                design = detect_design(llm, vs, fn, return_full=True)
            except Exception as e:
                status.warning(f"Could not classify {fn}: {e}")
                progress.progress((i + 1) / len(S.selected))
                continue
            S.detection_results[fn] = {
                "confidence": design.confidence,
                "rationale": design.rationale,
                "recommended": design.recommended_rubric,
            }
            # Auto-fill the rubric only if it's a real shipped rubric.
            if design.recommended_rubric in rubrics:
                S.paper_rubrics[fn] = design.recommended_rubric
            elif design.recommended_rubric in {"none", "unknown"}:
                # Don't auto-assign anything that doesn't fit a shipped rubric;
                # the user will see the rationale and decide whether to skip.
                S.paper_rubrics.pop(fn, None)
            progress.progress((i + 1) / len(S.selected))
        status.success("Done. Review the rationale and confidence below before grading.")
        time.sleep(0.6)
        st.rerun()

    # Per-paper rubric table with rationale + confidence columns.
    st.markdown("**Per-paper rubric assignment**")
    st.caption(
        "The Confidence and Rationale columns are populated by auto-detect. "
        "Rows with no shipped rubric (`none` recommendations from systematic reviews, "
        "diagnostic-accuracy studies, etc.) need a manual override or will be skipped."
    )
    df_rows = []
    for fn in S.selected:
        det = S.detection_results.get(fn, {})
        df_rows.append({
            "Paper": fn,
            "Rubric": S.paper_rubrics.get(fn, ""),
            "Confidence": det.get("confidence", ""),
            "Rationale": det.get("rationale", ""),
        })
    df = pd.DataFrame(df_rows)
    edited = st.data_editor(
        df,
        column_config={
            "Paper": st.column_config.TextColumn("Paper", disabled=True, width="medium"),
            "Rubric": st.column_config.SelectboxColumn(
                "Rubric", options=[""] + rubrics, width="small",
                help="Pick a rubric. Empty rows will be skipped.",
            ),
            "Confidence": st.column_config.NumberColumn(
                "Conf.", disabled=True, format="%.2f", width="small",
                help="Auto-detect confidence (0.00 - 1.00). Below 0.6 = manual review recommended.",
            ),
            "Rationale": st.column_config.TextColumn(
                "Rationale", disabled=True, width="large",
                help="Why auto-detect picked this rubric.",
            ),
        },
        hide_index=True,
        use_container_width=True,
    )
    for _, row in edited.iterrows():
        if row["Rubric"]:
            S.paper_rubrics[row["Paper"]] = row["Rubric"]
        else:
            S.paper_rubrics.pop(row["Paper"], None)

    st.divider()
    ready = [fn for fn in S.selected if fn in S.paper_rubrics]
    c1, c2 = st.columns([1, 5])
    with c1:
        if st.button("← Back"):
            S.step = "pick"
            st.rerun()
    with c2:
        if st.button(
            f"Run grading on {len(ready)} paper(s) →",
            type="primary",
            disabled=not ready,
        ):
            S.step = "grade"
            st.rerun()


# ============================================================
# STEP 3 — Run grading
# ============================================================

elif S.step == "grade":
    st.subheader("Running multi-run grading")
    env = _env()
    st.caption(
        f"{env['runs']} runs/domain at temperature {env['temperature']}. "
        f"Judgments below {int(env['agreement_threshold']*100)}% agreement will be flagged."
    )

    vs, _ = _vector_store()
    papers_to_grade = [fn for fn in S.selected if fn in S.paper_rubrics]

    if not papers_to_grade:
        st.error("No papers with assigned rubrics. Go back to step 2.")
        if st.button("← Back"):
            S.step = "rubric"
            st.rerun()
        st.stop()

    overall = st.progress(0.0)
    paper_status = st.empty()
    domain_status = st.empty()
    log = st.container(height=240, border=True)

    if st.button("▶ Start grading", type="primary"):
        for i, fn in enumerate(papers_to_grade):
            rubric_id = S.paper_rubrics[fn]
            try:
                rubric = load_rubric(rubric_id)
            except SystemExit as e:
                log.error(str(e))
                continue

            paper_status.info(
                f"**[{i+1}/{len(papers_to_grade)}]** Grading **{fn}** with `{rubric['name']}`"
            )
            domain_progress = st.progress(0.0)

            def cb(stage, payload):
                if stage == "domain_start":
                    domain_status.write(
                        f"  → {payload['domain']['id']}: {payload['domain']['name']} "
                        f"(running {env['runs']}×…)"
                    )
                elif stage == "domain_done":
                    d, r = payload["domain"], payload["result"]
                    flag = " 🚩" if r["flagged"] else ""
                    log.write(
                        f"`{fn}` · **{d['id']}** {d['name']} → "
                        f"**{r['final_judgment']}** "
                        f"({r['agreement_count']}/{env['runs']}){flag} "
                        f"in {payload['elapsed']:.1f}s"
                    )
                    domain_progress.progress((payload["index"] + 1) /
                                             max(1, len(rubric["domains"])))

            try:
                appraisal = grade_paper(env, vs, fn, rubric, progress_cb=cb)
                write_outputs(appraisal)
                S.appraisals[fn] = appraisal
            except Exception as e:
                log.error(f"`{fn}` failed: {e}")
            domain_progress.progress(1.0)
            overall.progress((i + 1) / len(papers_to_grade))

        paper_status.success(f"Graded {len(S.appraisals)} paper(s).")
        # Re-index appraisals so the Query page can search them.
        try:
            reindex_appraisals(env)
        except Exception as e:
            st.warning(f"Could not re-index appraisals: {e}")

        time.sleep(0.5)
        S.step = "review"
        st.rerun()

    st.caption(
        "Click **Start grading** to begin. You can leave this tab — progress "
        "is saved to `outputs/` after each paper completes."
    )


# ============================================================
# STEP 4 — Review & adjudicate
# ============================================================

elif S.step == "review":
    st.subheader("Review results")

    if not APPRAISALS_CSV.exists():
        st.warning("No appraisals on disk yet.")
        if st.button("← Back to grading"):
            S.step = "grade"
            st.rerun()
        st.stop()

    with APPRAISALS_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        st.warning("Empty appraisals.csv.")
        st.stop()

    rubrics_present = sorted({r["rubric"] for r in rows})
    rubric_filter = st.selectbox(
        "Show rubric",
        options=rubrics_present,
        index=0,
    )
    sub = [r for r in rows if r["rubric"] == rubric_filter]

    # --- Traffic-light plot ---
    st.markdown("### Traffic-light plot")
    try:
        fig = visualize_plot(sub, rubric_filter, out_path=None, return_fig=True)
        if fig is not None:
            st.pyplot(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Could not render plot: {e}")

    # --- Long-form table ---
    st.markdown("### Per-domain table")
    df = pd.DataFrame(sub)[
        ["filename", "domain_id", "domain_name", "judgment",
         "agreement", "flagged", "rationale"]
    ]
    st.dataframe(df, use_container_width=True, hide_index=True)

    # --- Adjudication for flagged judgments ---
    st.markdown("### 🚩 Adjudicate disagreements")
    flagged = [r for r in sub if r["flagged"] == "True"]
    if not flagged:
        st.success(
            "No flagged judgments — every domain met the agreement threshold. "
            "Nothing to adjudicate."
        )
    else:
        st.caption(
            f"{len(flagged)} judgment(s) below the agreement threshold. "
            "Review each one and pick the final rating as tie-breaker."
        )
        for r in flagged:
            key = (r["filename"], r["rubric"], r["domain_id"])
            with st.expander(
                f"**{r['filename']}** · {r['domain_id']} {r['domain_name']} → "
                f"{r['judgment']} ({r['agreement']})",
                expanded=False,
            ):
                # Show the run breakdown.
                st.markdown(f"**Inter-run breakdown:** `{r['runs_breakdown']}`")
                st.markdown(f"**LLM rationale:** {r['rationale']}")
                st.markdown("**Supporting quotes:**")
                if r["supporting_quotes"]:
                    for line in r["supporting_quotes"].split(" | "):
                        st.markdown(f"- {line}")
                else:
                    st.caption("(no verified quotes)")

                # Pull all per-run details from the JSON sidecar if available.
                base = r["filename"].rsplit(".", 1)[0]
                json_path = OUTPUTS_DIR / f"{base}.{r['rubric']}.appraisal.json"
                if json_path.exists():
                    appraisal = json.loads(json_path.read_text(encoding="utf-8"))
                    matching = next(
                        (d for d in appraisal["domains"] if d["domain_id"] == r["domain_id"]),
                        None,
                    )
                    if matching and matching.get("all_runs"):
                        st.markdown("**Per-run judgments:**")
                        run_df = pd.DataFrame([
                            {
                                "Run": run["run"],
                                "Judgment": run["judgment"],
                                "Quotes": " | ".join(
                                    f'"{ev["quote"][:100]}..."'
                                    if len(ev["quote"]) > 100 else f'"{ev["quote"]}"'
                                    for ev in run["evidence"]
                                ),
                            }
                            for run in matching["all_runs"]
                        ])
                        st.dataframe(run_df, use_container_width=True, hide_index=True)

                # Tie-breaker selector.
                rubric = load_rubric(r["rubric"])
                options = rubric["judgment_options"]
                current = S.adjudications.get(key, r["judgment"])
                final = st.radio(
                    "Final adjudicated judgment",
                    options=options,
                    index=options.index(current) if current in options else 0,
                    key=f"adj_{key}",
                    horizontal=True,
                )
                S.adjudications[key] = final
                st.caption(
                    f"Original LLM consensus: **{r['judgment']}** · "
                    f"Your tie-breaker: **{final}**"
                )

        # --- Save adjudicated CSV ---
        if st.button("💾 Save adjudicated CSV", type="primary"):
            out_path = OUTPUTS_DIR / f"appraisals_{rubric_filter}_adjudicated.csv"
            adj_rows = []
            for r in sub:
                key = (r["filename"], r["rubric"], r["domain_id"])
                final = S.adjudications.get(key, r["judgment"])
                adj_rows.append({
                    **r,
                    "final_adjudicated": final,
                    "adjudication_changed": "yes" if final != r["judgment"] else "no",
                })
            with out_path.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(adj_rows[0].keys()))
                w.writeheader()
                w.writerows(adj_rows)
            st.success(f"Saved {out_path}")
            with out_path.open("rb") as f:
                st.download_button(
                    "⬇ Download adjudicated CSV",
                    data=f.read(),
                    file_name=out_path.name,
                    mime="text/csv",
                )

    st.divider()
    c1, c2 = st.columns([1, 5])
    with c1:
        if st.button("← Grade more"):
            S.step = "pick"
            st.rerun()
    with c2:
        if st.button("Reset workflow"):
            for k in ["step", "selected", "paper_rubrics", "appraisals", "adjudications"]:
                S.pop(k, None)
            st.rerun()
