"""Body-of-evidence GRADE assessment for a specific outcome across the corpus.

GRADE differs from study-level appraisal: it rates the *certainty of evidence*
for one outcome across many studies. The pipeline:

    1. User specifies the outcome (e.g., "mortality after burn injury")
    2. We retrieve cross-corpus chunks discussing that outcome
    3. We pull existing per-study RoB judgments from outputs/appraisals.csv
    4. We assess the 5 downgrade + 3 upgrade GRADE criteria with multi-run LLM
    5. Compute final certainty (HIGH / MODERATE / LOW / VERY LOW)

Every judgment is grounded: each cited quote is verified against the retrieved
chunks (so the page number comes from the index, not the LLM).

Usage:
    py grade.py --outcome "mortality" --start-from rct
    py grade.py --outcome "length of hospital stay" --start-from non_rct
    py grade.py --outcome "infection rate" --runs 5
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure") and getattr(_stream, "encoding", "").lower() != "utf-8":
        _stream.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from appraise import (
    APPRAISALS_CSV,
    DomainJudgment,
    OUTPUTS_DIR,
    RUBRICS_DIR,
    _coerce_judgment,
    _normalize,
    format_context,
    locate_quote,
)


def get_env():
    load_dotenv()
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


def load_grade_rubric() -> dict:
    return json.loads((RUBRICS_DIR / "grade.json").read_text(encoding="utf-8"))


def retrieve_for_outcome(vector_store: QdrantVectorStore, outcome: str,
                         queries: list[str], k: int) -> list[Document]:
    """Cross-corpus retrieval. Combine the outcome with each domain query for focus."""
    seen: dict[tuple, Document] = {}
    for q in queries:
        full_q = f"{outcome}: {q}"
        for d in vector_store.similarity_search(full_q, k=k):
            key = (d.metadata.get("source"), d.metadata.get("page"), d.page_content[:80])
            seen.setdefault(key, d)
    return list(seen.values())


def existing_rob_summary(outcome: str) -> str:
    """Summarize study-level RoB judgments from appraisals.csv. This feeds the
    G1 (Risk of Bias) domain so GRADE can lean on the per-study work already done."""
    if not APPRAISALS_CSV.exists():
        return "(No prior study-level appraisals available.)"
    with APPRAISALS_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return "(appraisals.csv is empty.)"
    # Aggregate per-paper overall judgment by taking the worst domain rating.
    severity = {
        "low": 0, "low risk of bias": 0, "no serious concern": 0, "1 star": 0, "2 stars": 0,
        "some concerns": 1, "moderate": 1,
        "serious": 2, "high": 3, "critical": 4, "0 stars": 3,
    }
    by_paper: dict[str, list[tuple[int, str]]] = {}
    for r in rows:
        sev = severity.get(r["judgment"].lower(), 1)
        by_paper.setdefault(r["filename"], []).append((sev, r["judgment"]))
    lines = []
    for fn, judgments in sorted(by_paper.items()):
        worst_sev, worst_label = max(judgments, key=lambda x: x[0])
        lines.append(f"  - {fn}: worst-domain rating = {worst_label}")
    return f"Per-study RoB (worst domain across {len(by_paper)} papers):\n" + "\n".join(lines)


def grade_one_domain(llm: ChatOpenAI, rubric: dict, domain: dict,
                     vector_store: QdrantVectorStore, outcome: str,
                     rob_summary: str, runs: int, k: int) -> dict:
    docs = retrieve_for_outcome(vector_store, outcome, domain["retrieval_queries"], k)
    context = format_context(docs)
    options = rubric["judgment_options"] if domain["kind"] == "downgrade" else rubric["upgrade_options"]

    extra = ""
    if domain["id"] == "G1":
        extra = f"\n\nRelevant per-study Risk of Bias context:\n{rob_summary}\n"

    sys_prompt = (
        f"You are applying GRADE to assess the body of evidence for the outcome: '{outcome}'.\n"
        f"You are judging the {'downgrade' if domain['kind']=='downgrade' else 'upgrade'} domain: {domain['name']}.\n\n"
        f"Choose ONE judgment from: {options}.\n\n"
        f"Domain criteria:\n{domain['criteria_summary']}\n\n"
        f"Signaling questions:\n" + "\n".join(f"- {q}" for q in domain['signaling_questions']) +
        extra +
        "\n\nRules:\n"
        "1. Use ONLY the excerpts and the per-study RoB context above. No outside knowledge.\n"
        "2. Each piece of evidence MUST be a verbatim quote from the excerpts (or refer "
        "to a per-study RoB judgment from the summary).\n"
        "3. Rationale must reference the specific quotes and/or per-study judgments used.\n"
    )
    user_prompt = f"Outcome: {outcome}\n\nExcerpts (cross-corpus):\n{context}"

    structured = llm.with_structured_output(DomainJudgment, method="function_calling")
    all_runs = []
    for i in range(runs):
        try:
            res = structured.invoke([("system", sys_prompt), ("human", user_prompt)])
        except Exception as e:
            print(f"      run {i+1}/{runs} failed: {e}", file=sys.stderr)
            continue
        verified = []
        for ev in res.evidence:
            page, ok = locate_quote(ev.quote, docs)
            verified.append({
                "quote": ev.quote, "supports": ev.supports,
                "page": page, "verified": ok,
                # If the page lookup failed AND the quote mentions a filename,
                # we may be citing a per-study RoB summary line — that's allowed.
            })
        judgment = res.judgment if res.judgment in options else _coerce_judgment(res.judgment, options)
        all_runs.append({
            "run": i + 1, "judgment": judgment, "evidence": verified,
            "rationale": res.rationale, "criteria_invoked": res.criteria_invoked,
        })

    if not all_runs:
        raise RuntimeError(f"All runs failed for {domain['id']}")

    counter = Counter(r["judgment"] for r in all_runs)
    final, count = counter.most_common(1)[0]
    winners = [r for r in all_runs if r["judgment"] == final]
    pooled = {}
    for r in winners:
        for ev in r["evidence"]:
            key = _normalize(ev["quote"])[:200]
            if key and key not in pooled:
                pooled[key] = ev
    rep = max(winners, key=lambda r: len(r["evidence"]))

    return {
        "domain_id": domain["id"], "domain_name": domain["name"], "kind": domain["kind"],
        "final_judgment": final,
        "agreement_count": count, "agreement_pct": count / len(all_runs),
        "evidence": list(pooled.values()),
        "rationale": rep["rationale"], "criteria_invoked": rep["criteria_invoked"],
        "runs_breakdown": dict(counter), "all_runs": all_runs,
        "source_documents": sorted({d.metadata.get("source") for d in docs if d.metadata.get("source")}),
    }


# Map judgment labels to integer adjustments.
_DOWNGRADE_DELTA = {
    "No serious concern": 0,
    "Serious (-1)": -1,
    "Very serious (-2)": -2,
}
_UPGRADE_DELTA = {
    "Not applicable": 0,
    "Large effect (+1)": 1,
    "Very large effect (+2)": 2,
    "Dose-response (+1)": 1,
    "Plausible confounders reduce effect (+1)": 1,
}
_LEVELS = ["Very low", "Low", "Moderate", "High"]  # index = certainty score


def compute_certainty(start: str, downgrades: list[int], upgrades: list[int]) -> str:
    score = _LEVELS.index(start)
    score += sum(downgrades)
    score += sum(upgrades)
    score = max(0, min(len(_LEVELS) - 1, score))
    return _LEVELS[score]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40]


def write_outputs(grade_result: dict):
    OUTPUTS_DIR.mkdir(exist_ok=True)
    slug = _slug(grade_result["outcome"])
    json_path = OUTPUTS_DIR / f"grade_{slug}.json"
    csv_path = OUTPUTS_DIR / f"grade_{slug}.csv"
    json_path.write_text(json.dumps(grade_result, indent=2), encoding="utf-8")

    fields = ["outcome", "domain_id", "domain_name", "kind", "judgment",
              "agreement", "supporting_quotes", "pages", "source_papers",
              "rationale", "criteria_invoked"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for d in grade_result["domains"]:
            quotes = " | ".join(
                f'"{ev["quote"][:200]}{"..." if len(ev["quote"])>200 else ""}" '
                f'(p.{ev.get("page","?")}{"" if ev.get("verified") else " [unverified]"})'
                for ev in d["evidence"]
            )
            pages = ", ".join(sorted({str(ev["page"]) for ev in d["evidence"] if ev.get("page") is not None}))
            w.writerow({
                "outcome": grade_result["outcome"],
                "domain_id": d["domain_id"], "domain_name": d["domain_name"],
                "kind": d["kind"], "judgment": d["final_judgment"],
                "agreement": f"{d['agreement_count']}/{grade_result['runs_per_domain']}",
                "supporting_quotes": quotes,
                "pages": pages,
                "source_papers": "; ".join(d["source_documents"]),
                "rationale": d["rationale"],
                "criteria_invoked": "; ".join(d["criteria_invoked"]),
            })
        # Final certainty as the closing row so the CSV is self-contained.
        w.writerow({
            "outcome": grade_result["outcome"],
            "domain_id": "FINAL",
            "domain_name": "Overall certainty",
            "kind": "summary",
            "judgment": grade_result["final_certainty"],
            "agreement": "",
            "supporting_quotes": "",
            "pages": "",
            "source_papers": "",
            "rationale": grade_result["certainty_rationale"],
            "criteria_invoked": "",
        })
    print(f"\nWrote {json_path.name} + {csv_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Body-of-evidence GRADE for one outcome.")
    parser.add_argument("--outcome", required=True, help='The outcome to grade (e.g., "mortality").')
    parser.add_argument("--start-from", choices=["rct", "non_rct"], default="rct",
                        help="Starting certainty: HIGH for RCT bodies, LOW for non-RCT bodies.")
    parser.add_argument("--runs", type=int, help="Override APPRAISE_RUNS for this run.")
    args = parser.parse_args()

    env = get_env()
    if args.runs:
        env["runs"] = args.runs
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set.")

    rubric = load_grade_rubric()
    client = QdrantClient(path=env["qdrant_path"])
    if not client.collection_exists(env["collection"]):
        sys.exit(f"Collection '{env['collection']}' not found. Run `py ingest.py` first.")

    embeddings = OpenAIEmbeddings(model=env["embed_model"])
    vector_store = QdrantVectorStore(
        client=client, collection_name=env["collection"], embedding=embeddings
    )
    llm = ChatOpenAI(model=env["llm_model"], temperature=env["temperature"])

    rob_summary = existing_rob_summary(args.outcome)
    print(f"\nGRADE — outcome: '{args.outcome}'  (start: {args.start_from.upper()})")
    print(f"Runs/domain: {env['runs']}, k={env['retrieve_k']}\n")

    domain_results = []
    for d in rubric["domains"]:
        print(f"  [{d['id']}] {d['name']} ({d['kind']}) — {env['runs']}x...")
        t0 = time.time()
        result = grade_one_domain(llm, rubric, d, vector_store, args.outcome,
                                  rob_summary, runs=env["runs"], k=env["retrieve_k"])
        result["flagged"] = result["agreement_pct"] < env["agreement_threshold"]
        domain_results.append(result)
        print(f"      → {result['final_judgment']} ({result['agreement_count']}/{env['runs']}) "
              f"in {time.time()-t0:.1f}s")

    # Compute final certainty.
    start = rubric["starting_certainty"]["rct" if args.start_from == "rct" else "non_rct"]
    downgrades = [_DOWNGRADE_DELTA.get(d["final_judgment"], 0)
                  for d in domain_results if d["kind"] == "downgrade"]
    upgrades = [_UPGRADE_DELTA.get(d["final_judgment"], 0)
                for d in domain_results if d["kind"] == "upgrade"]
    final = compute_certainty(start, downgrades, upgrades)

    # Build a brief rationale that lists the deltas.
    parts = [f"Started at {start} ({'RCT' if args.start_from=='rct' else 'non-RCT'} body)."]
    for d in domain_results:
        delta = (_DOWNGRADE_DELTA if d["kind"] == "downgrade" else _UPGRADE_DELTA).get(d["final_judgment"], 0)
        if delta:
            parts.append(f"{d['name']}: {d['final_judgment']} ({delta:+d}).")
    parts.append(f"Final certainty: {final}.")
    cert_rationale = " ".join(parts)

    grade_result = {
        "outcome": args.outcome,
        "started_from": start,
        "starting_design": args.start_from,
        "runs_per_domain": env["runs"],
        "agreement_threshold": env["agreement_threshold"],
        "graded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "domains": domain_results,
        "final_certainty": final,
        "certainty_rationale": cert_rationale,
    }
    print(f"\n=== FINAL CERTAINTY: {final} ===")
    print(cert_rationale)
    write_outputs(grade_result)


if __name__ == "__main__":
    main()
