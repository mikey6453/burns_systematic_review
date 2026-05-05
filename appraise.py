"""Appraise PDFs in the corpus against an evidence-grading rubric.

Multi-run grading: each domain is judged N times at non-zero temperature so we
can measure inter-run agreement. The modal judgment becomes the final rating;
disagreements below a configurable threshold are flagged for human review.

Every judgment is traceable: the LLM must return verbatim quotes from the
retrieved chunks. Each quote is matched back to its source chunk to recover the
filename + page number (so the page is grounded in the index, not in the LLM).

Usage:
    py appraise.py --paper foo.pdf --rubric rob2
    py appraise.py --all                              # sweep every paper, auto-detect rubric
    py appraise.py --all --rubric rob2                # force a rubric on all papers
    py appraise.py --paper foo.pdf --dry-run          # estimate cost, run nothing
    py appraise.py --reindex                          # (re)build burns_appraisals from outputs/
    py appraise.py --list                             # show papers in the corpus

Outputs:
    outputs/<paper>.appraisal.json     # full per-run detail
    outputs/appraisals.csv             # long format: one row per (paper, domain)
    outputs/appraisals_summary.csv     # wide format: one row per paper
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

# Make stdout UTF-8 so Unicode (em-dashes, arrows, accented filenames) prints
# cleanly on Windows where the default console codec is cp1252.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure") and getattr(_stream, "encoding", "").lower() != "utf-8":
        _stream.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    VectorParams,
)

ROOT = Path(__file__).parent
RUBRICS_DIR = ROOT / "rubrics"
OUTPUTS_DIR = ROOT / "outputs"
APPRAISALS_CSV = OUTPUTS_DIR / "appraisals.csv"
SUMMARY_CSV = OUTPUTS_DIR / "appraisals_summary.csv"

CSV_FIELDS = [
    "filename", "rubric", "domain_id", "domain_name",
    "judgment", "agreement", "agreement_pct", "flagged",
    "supporting_quotes", "pages", "rationale",
    "criteria_invoked", "runs_breakdown",
]


# --- Pydantic schemas for structured LLM output ---

class EvidenceItem(BaseModel):
    quote: str = Field(description="A verbatim sentence or short passage copied EXACTLY from the provided excerpts.")
    # Optional + default so an occasional omission by the LLM doesn't kill the
    # whole run. The 'quote' is what we actually verify against the chunks.
    supports: str = Field(
        default="",
        description="Which signaling question or rubric criterion this quote addresses.",
    )


class DomainJudgment(BaseModel):
    judgment: str = Field(description="One of the rubric's allowed judgment options.")
    evidence: list[EvidenceItem] = Field(
        description="At least one verbatim quote from the excerpts. If the paper truly provides no information, return an empty list and set no_information=true."
    )
    rationale: str = Field(
        description="2-4 sentence rationale that explicitly references the cited quotes."
    )
    criteria_invoked: list[str] = Field(
        description="Names of the specific rubric criteria or signaling questions used to reach this judgment."
    )
    no_information: bool = Field(
        default=False,
        description="True only if the excerpts contain no usable information for this domain.",
    )


# --- Setup ---

def get_env():
    load_dotenv()
    return {
        "qdrant_path": os.environ.get("QDRANT_PATH", "./qdrant_data"),
        "collection": os.environ.get("COLLECTION_NAME", "burns_papers"),
        "appraisals_collection": os.environ.get("APPRAISALS_COLLECTION", "burns_appraisals"),
        "embed_model": os.environ.get("EMBED_MODEL", "text-embedding-3-small"),
        "llm_model": os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        "runs": int(os.environ.get("APPRAISE_RUNS", "10")),
        "temperature": float(os.environ.get("APPRAISE_TEMPERATURE", "0.5")),
        "agreement_threshold": float(os.environ.get("AGREEMENT_THRESHOLD", "0.7")),
        "retrieve_k": int(os.environ.get("APPRAISE_K", "6")),
    }


def load_rubric(rubric_id: str) -> dict:
    path = RUBRICS_DIR / f"{rubric_id}.json"
    if not path.exists():
        sys.exit(f"Rubric not found: {path}. Available: {[p.stem for p in RUBRICS_DIR.glob('*.json')]}")
    rubric = json.loads(path.read_text(encoding="utf-8"))
    if rubric.get("level") == "outcome":
        sys.exit(f"Rubric '{rubric_id}' is body-of-evidence, not study-level. Use grade.py instead.")
    return rubric


# --- Corpus introspection ---

def list_papers(client: QdrantClient, collection: str) -> list[str]:
    """Enumerate unique source filenames in the PDF collection."""
    if not client.collection_exists(collection):
        sys.exit(f"Collection '{collection}' not found. Run `py ingest.py` first.")
    seen = set()
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            limit=1000,
            offset=next_offset,
            with_payload=True,
        )
        for p in points:
            md = (p.payload or {}).get("metadata", {})
            src = md.get("source")
            if src:
                seen.add(src)
        if next_offset is None:
            break
    return sorted(seen)


# --- Retrieval ---

def retrieve_for_domain(vector_store: QdrantVectorStore, filename: str,
                        queries: list[str], k: int) -> list[Document]:
    """Run multiple targeted queries scoped to one paper, dedupe by chunk text."""
    flt = Filter(must=[
        FieldCondition(key="metadata.source", match=MatchValue(value=filename))
    ])
    seen: dict[tuple, Document] = {}
    for q in queries:
        for d in vector_store.similarity_search(q, k=k, filter=flt):
            key = (d.metadata.get("source"), d.metadata.get("page"), d.page_content[:80])
            seen.setdefault(key, d)
    return list(seen.values())


def format_context(docs: list[Document]) -> str:
    """Render chunks with citation tags the LLM can copy into its quotes."""
    return "\n\n---\n\n".join(
        f"[{d.metadata.get('source', '?')} p.{d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )


# --- Evidence verification (the traceability guarantee) ---

_WS = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WS.sub(" ", text.lower()).strip()


def locate_quote(quote: str, docs: list[Document]) -> tuple[int | None, bool]:
    """Find which retrieved chunk contains the quote. Returns (page_or_None, verified).

    Verified=True means the quote appears verbatim (modulo whitespace) in some
    retrieved chunk, and the page is taken from that chunk's metadata. If the
    quote can't be found, page is None and verified=False — flagging a possible
    hallucination.
    """
    if not quote.strip():
        return None, False
    needle = _normalize(quote)
    for d in docs:
        if needle in _normalize(d.page_content):
            return d.metadata.get("page"), True
    return None, False


# --- Single-domain multi-run grading ---

def grade_domain(llm: ChatOpenAI, rubric: dict, domain: dict,
                 vector_store: QdrantVectorStore, filename: str,
                 runs: int, k: int) -> dict:
    docs = retrieve_for_domain(vector_store, filename, domain["retrieval_queries"], k)
    if not docs:
        return {
            "domain_id": domain["id"],
            "domain_name": domain["name"],
            "final_judgment": rubric.get("no_information_label", "No information"),
            "agreement_count": 0,
            "agreement_pct": 0.0,
            "flagged": True,
            "evidence": [],
            "rationale": "No chunks retrieved for this domain — paper may not address it.",
            "criteria_invoked": [],
            "runs_breakdown": {},
            "all_runs": [],
        }

    context = format_context(docs)
    judgment_options = rubric["judgment_options"]
    sys_prompt = (
        f"You are an evidence appraiser applying the {rubric['name']} rubric.\n"
        f"You are judging the domain: {domain['name']}.\n\n"
        f"Choose ONE judgment from: {judgment_options}.\n\n"
        f"Domain criteria:\n{domain.get('criteria_summary', '')}\n\n"
        f"Signaling questions:\n" + "\n".join(f"- {q}" for q in domain["signaling_questions"]) +
        "\n\nRules:\n"
        "1. Use ONLY the excerpts below. Do not rely on outside knowledge.\n"
        "2. Each piece of evidence must be a VERBATIM quote copied from the excerpts. "
        "Do not paraphrase, do not add ellipses, do not invent.\n"
        "3. Your rationale must explicitly tie each criterion to the quotes.\n"
        "4. If the excerpts truly do not address this domain, set no_information=true and choose "
        f"the judgment '{rubric.get('no_information_label', judgment_options[-1])}'.\n"
    )
    user_prompt = f"Paper: {filename}\n\nExcerpts:\n{context}"

    structured = llm.with_structured_output(DomainJudgment, method="function_calling")

    all_runs = []
    for i in range(runs):
        try:
            res: DomainJudgment = structured.invoke([("system", sys_prompt), ("human", user_prompt)])
        except Exception as e:
            print(f"      run {i+1}/{runs} failed: {e}", file=sys.stderr)
            continue
        # Verify each quote against the retrieved context, attach grounded page.
        verified_evidence = []
        for ev in res.evidence:
            page, verified = locate_quote(ev.quote, docs)
            verified_evidence.append({
                "quote": ev.quote,
                "supports": ev.supports,
                "page": page,
                "verified": verified,
            })
        # Coerce judgment into an allowed option (LLM occasionally drifts).
        judgment = res.judgment if res.judgment in judgment_options else _coerce_judgment(res.judgment, judgment_options)
        all_runs.append({
            "run": i + 1,
            "judgment": judgment,
            "no_information": res.no_information,
            "evidence": verified_evidence,
            "rationale": res.rationale,
            "criteria_invoked": res.criteria_invoked,
        })

    if not all_runs:
        raise RuntimeError(f"All {runs} runs failed for domain {domain['id']}")

    # Aggregate.
    counter = Counter(r["judgment"] for r in all_runs)
    final, count = counter.most_common(1)[0]
    pct = count / len(all_runs)

    # Pool evidence from runs that voted with the majority — these are what
    # actually drove the final decision. Dedupe by normalized quote text.
    winning_runs = [r for r in all_runs if r["judgment"] == final]
    pooled_evidence: dict[str, dict] = {}
    for r in winning_runs:
        for ev in r["evidence"]:
            key = _normalize(ev["quote"])[:200]
            if key and key not in pooled_evidence:
                pooled_evidence[key] = ev

    # Use the rationale + criteria from the highest-evidence winning run.
    representative = max(winning_runs, key=lambda r: len(r["evidence"]))

    return {
        "domain_id": domain["id"],
        "domain_name": domain["name"],
        "final_judgment": final,
        "agreement_count": count,
        "agreement_pct": pct,
        "flagged": pct < FLAG_THRESHOLD_PLACEHOLDER,  # set by caller, see grade_paper
        "evidence": list(pooled_evidence.values()),
        "rationale": representative["rationale"],
        "criteria_invoked": representative["criteria_invoked"],
        "runs_breakdown": dict(counter),
        "all_runs": all_runs,
    }


FLAG_THRESHOLD_PLACEHOLDER = 0.7  # overwritten via env at runtime


def _coerce_judgment(raw: str, options: list[str]) -> str:
    """If the LLM returns a slightly off label, pick the closest exact-prefix match."""
    raw_n = raw.strip().lower()
    for opt in options:
        if raw_n == opt.lower() or raw_n.startswith(opt.lower()):
            return opt
    for opt in options:
        if opt.lower() in raw_n:
            return opt
    return raw  # surface the raw label so downstream sees the drift


# --- Whole-paper appraisal ---

def detect_design(llm: ChatOpenAI, vector_store: QdrantVectorStore, filename: str) -> str:
    """Classify the paper and recommend a rubric id."""
    docs = retrieve_for_domain(
        vector_store, filename,
        ["abstract objectives", "methods study design population", "randomized cohort observational"],
        k=4,
    )
    if not docs:
        return "unknown"
    context = format_context(docs)
    prompt = (
        "Classify the study design from these excerpts. Reply with ONE token only:\n"
        "  rob2       -> randomized controlled trial\n"
        "  robins_i   -> non-randomized intervention study\n"
        "  nos_cohort -> observational cohort or case-control\n"
        "  unknown    -> cannot determine\n\n"
        f"Excerpts:\n{context}\n\nReply with one token."
    )
    resp = llm.invoke(prompt).content.strip().lower().split()[0]
    return resp if resp in {"rob2", "robins_i", "nos_cohort"} else "unknown"


def grade_paper(env: dict, vector_store: QdrantVectorStore,
                filename: str, rubric: dict) -> dict:
    print(f"\n=== {filename} — {rubric['name']} ===")
    runs = env["runs"]
    k = env["retrieve_k"]
    threshold = env["agreement_threshold"]

    # Per-run LLM at non-zero temperature for genuine sampling diversity.
    llm = ChatOpenAI(model=env["llm_model"], temperature=env["temperature"])

    domain_results = []
    for d in rubric["domains"]:
        print(f"  [{d['id']}] {d['name']} — running {runs}x...")
        t0 = time.time()
        result = grade_domain(llm, rubric, d, vector_store, filename, runs=runs, k=k)
        result["flagged"] = result["agreement_pct"] < threshold
        domain_results.append(result)
        print(f"      → {result['final_judgment']} ({result['agreement_count']}/{runs}, "
              f"{'FLAGGED' if result['flagged'] else 'ok'}) in {time.time()-t0:.1f}s")

    return {
        "filename": filename,
        "rubric": rubric["id"],
        "rubric_name": rubric["name"],
        "graded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runs_per_domain": runs,
        "agreement_threshold": threshold,
        "domains": domain_results,
    }


# --- Output: JSON + CSV ---

def _quotes_summary(evidence: list[dict]) -> tuple[str, str]:
    """Render evidence as compact CSV cells: quotes-with-pages and pages-only."""
    parts = []
    pages = []
    for ev in evidence:
        page = ev.get("page")
        verified = ev.get("verified", False)
        suffix = f" (p.{page})" if page is not None else " (p.?)"
        if not verified:
            suffix += " [unverified]"
        # Truncate very long quotes for the CSV cell.
        quote = ev["quote"][:300] + ("..." if len(ev["quote"]) > 300 else "")
        parts.append(f'"{quote}"{suffix}')
        if page is not None and page not in pages:
            pages.append(page)
    return " | ".join(parts), ", ".join(str(p) for p in pages)


def write_outputs(appraisal: dict):
    OUTPUTS_DIR.mkdir(exist_ok=True)
    base = appraisal["filename"].rsplit(".", 1)[0]

    # 1) Per-paper JSON (full audit trail incl. all runs).
    json_path = OUTPUTS_DIR / f"{base}.{appraisal['rubric']}.appraisal.json"
    json_path.write_text(json.dumps(appraisal, indent=2), encoding="utf-8")

    # 2) Long-format CSV (one row per domain, append-only).
    rows = []
    for d in appraisal["domains"]:
        quotes_str, pages_str = _quotes_summary(d["evidence"])
        rows.append({
            "filename": appraisal["filename"],
            "rubric": appraisal["rubric"],
            "domain_id": d["domain_id"],
            "domain_name": d["domain_name"],
            "judgment": d["final_judgment"],
            "agreement": f"{d['agreement_count']}/{appraisal['runs_per_domain']}",
            "agreement_pct": f"{d['agreement_pct']:.2f}",
            "flagged": d["flagged"],
            "supporting_quotes": quotes_str,
            "pages": pages_str,
            "rationale": d["rationale"],
            "criteria_invoked": "; ".join(d["criteria_invoked"]),
            "runs_breakdown": "; ".join(f"{k}:{v}" for k, v in d["runs_breakdown"].items()),
        })

    # Dedupe by (filename, rubric, domain_id) when appending — re-running a
    # paper should overwrite the old rows, not stack new ones.
    existing = []
    if APPRAISALS_CSV.exists():
        with APPRAISALS_CSV.open(encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    keep = [
        r for r in existing
        if (r["filename"], r["rubric"], r["domain_id"]) not in
           {(rr["filename"], rr["rubric"], rr["domain_id"]) for rr in rows}
    ]
    with APPRAISALS_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in keep + rows:
            w.writerow(r)

    # 3) Wide-format summary (one row per (paper, rubric)).
    rebuild_summary_csv()

    print(f"  wrote {json_path.name}, updated appraisals.csv + appraisals_summary.csv")


def rebuild_summary_csv():
    """Regenerate the wide CSV from the long CSV. Cheap; called after each paper."""
    if not APPRAISALS_CSV.exists():
        return
    with APPRAISALS_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    # Group by (filename, rubric); columns are the union of domain_ids per rubric.
    by_paper: dict[tuple[str, str], dict] = {}
    domain_cols: dict[str, list[str]] = {}
    for r in rows:
        key = (r["filename"], r["rubric"])
        by_paper.setdefault(key, {})[r["domain_id"]] = r
        domain_cols.setdefault(r["rubric"], [])
        if r["domain_id"] not in domain_cols[r["rubric"]]:
            domain_cols[r["rubric"]].append(r["domain_id"])
    # Stable column order per rubric; one summary CSV with all rubrics.
    fieldnames = ["filename", "rubric", "any_flagged"]
    seen_dom = []
    for rubric_id, doms in domain_cols.items():
        for d in doms:
            tag = f"{rubric_id}:{d}"
            if tag not in seen_dom:
                seen_dom.append(tag)
                fieldnames.append(tag)
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for (fn, rb), domains in sorted(by_paper.items()):
            row = {"filename": fn, "rubric": rb,
                   "any_flagged": any(d["flagged"] == "True" for d in domains.values())}
            for tag in seen_dom:
                rb2, did = tag.split(":", 1)
                if rb2 == rb and did in domains:
                    d = domains[did]
                    row[tag] = f"{d['judgment']} ({d['agreement']})"
                else:
                    row[tag] = ""
            w.writerow(row)


# --- Re-indexing appraisals as a queryable Qdrant collection ---

def reindex_appraisals(env: dict):
    """Embed each appraisal row as a document so query.py can ask questions
    over judgments themselves (e.g., 'which RCTs have low risk for D2?')."""
    if not APPRAISALS_CSV.exists():
        sys.exit(f"No appraisals to index — {APPRAISALS_CSV} not found. Run appraise.py first.")

    with APPRAISALS_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit("appraisals.csv is empty.")

    print(f"Re-indexing {len(rows)} appraisal rows into '{env['appraisals_collection']}'...")
    client = QdrantClient(path=env["qdrant_path"])
    if client.collection_exists(env["appraisals_collection"]):
        client.delete_collection(env["appraisals_collection"])
    client.create_collection(
        collection_name=env["appraisals_collection"],
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    embeddings = OpenAIEmbeddings(model=env["embed_model"])
    vector_store = QdrantVectorStore(
        client=client, collection_name=env["appraisals_collection"], embedding=embeddings
    )
    docs = []
    for r in rows:
        # The text we embed is what users will semantically search against.
        # Putting domain name + judgment + rationale + quotes here makes
        # natural-language queries like "low risk randomization" work.
        text = (
            f"Paper: {r['filename']}\n"
            f"Rubric: {r['rubric']} — {r['domain_id']} {r['domain_name']}\n"
            f"Judgment: {r['judgment']} (agreement {r['agreement']}, "
            f"{'flagged' if r['flagged']=='True' else 'not flagged'})\n"
            f"Rationale: {r['rationale']}\n"
            f"Supporting evidence: {r['supporting_quotes']}\n"
            f"Criteria invoked: {r['criteria_invoked']}"
        )
        # Metadata mirrors PDF metadata (source/page) so query.py treats
        # appraisal rows like any other citable document.
        first_page = r["pages"].split(",")[0].strip() if r["pages"] else ""
        meta = {
            "source": r["filename"],
            "page": int(first_page) if first_page.isdigit() else 0,
            "rubric": r["rubric"],
            "domain_id": r["domain_id"],
            "judgment": r["judgment"],
            "flagged": r["flagged"],
            "kind": "appraisal",
        }
        docs.append(Document(page_content=text, metadata=meta))
    vector_store.add_documents(docs)
    print(f"Indexed {len(docs)} judgments. Query with: py query.py --collection appraisals")


# --- Cost estimation (--dry-run) ---

def estimate_cost(env: dict, paper_count: int, rubric: dict) -> float:
    """Rough $ estimate for gpt-4o-mini @ public pricing. Order-of-magnitude only."""
    domains = len(rubric["domains"])
    runs = env["runs"]
    # ~3000 input tokens (system + 6 chunks of context), ~400 output tokens per call.
    input_tokens = paper_count * domains * runs * 3000
    output_tokens = paper_count * domains * runs * 400
    # gpt-4o-mini: $0.15 / 1M input, $0.60 / 1M output (as of 2024-2025).
    cost = input_tokens * 0.15 / 1_000_000 + output_tokens * 0.60 / 1_000_000
    return cost


# --- CLI ---

def main():
    parser = argparse.ArgumentParser(description="Appraise PDFs against an evidence rubric.")
    parser.add_argument("--paper", help="Filename in the corpus (e.g., iraa103.pdf).")
    parser.add_argument("--rubric", help="Rubric id (rob2, robins_i, nos_cohort, custom). If omitted, auto-detected.")
    parser.add_argument("--all", action="store_true", help="Grade every paper in the corpus.")
    parser.add_argument("--list", action="store_true", help="List papers in the corpus and exit.")
    parser.add_argument("--reindex", action="store_true", help="(Re)build the appraisals Qdrant collection from CSV.")
    parser.add_argument("--dry-run", action="store_true", help="Print cost estimate, run nothing.")
    args = parser.parse_args()

    env = get_env()
    client = QdrantClient(path=env["qdrant_path"])

    if args.list:
        for p in list_papers(client, env["collection"]):
            print(p)
        return

    if args.reindex:
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("OPENAI_API_KEY not set (needed for embeddings). Copy .env.example to .env.")
        reindex_appraisals(env)
        return

    # Resolve which papers + which rubric (no OpenAI needed up to here).
    papers = list_papers(client, env["collection"])
    if args.all:
        target_papers = papers
    elif args.paper:
        if args.paper not in papers:
            sys.exit(f"'{args.paper}' not in corpus. Use --list to see available papers.")
        target_papers = [args.paper]
    else:
        sys.exit("Specify --paper, --all, --list, or --reindex.")

    # If rubric not specified, we'll auto-detect per paper later.
    fixed_rubric = load_rubric(args.rubric) if args.rubric else None

    if args.dry_run:
        # Use the fixed rubric or the largest available for an upper bound.
        sample = fixed_rubric or load_rubric("robins_i")
        cost = estimate_cost(env, len(target_papers), sample)
        print(f"Papers: {len(target_papers)}, runs/domain: {env['runs']}, "
              f"domains: {len(sample['domains'])}")
        print(f"Estimated cost (gpt-4o-mini): ${cost:.2f}")
        print("(Order-of-magnitude only — actual depends on retrieved chunk size.)")
        return

    # Past this point we need OpenAI: embeddings for retrieval, LLM for grading.
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set. Copy .env.example to .env and fill it in.")

    embeddings = OpenAIEmbeddings(model=env["embed_model"])
    vector_store = QdrantVectorStore(
        client=client, collection_name=env["collection"], embedding=embeddings
    )
    detect_llm = ChatOpenAI(model=env["llm_model"], temperature=0)

    for i, paper in enumerate(target_papers, 1):
        print(f"\n[{i}/{len(target_papers)}] {paper}")
        rubric = fixed_rubric
        if rubric is None:
            recommended = detect_design(detect_llm, vector_store, paper)
            if recommended == "unknown":
                print(f"  could not auto-detect a rubric; skipping. Use --rubric to force.")
                continue
            print(f"  auto-detected: {recommended}")
            rubric = load_rubric(recommended)
        try:
            appraisal = grade_paper(env, vector_store, paper, rubric)
            write_outputs(appraisal)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Auto re-index after a sweep so the appraisals collection stays fresh.
    if args.all and APPRAISALS_CSV.exists():
        print("\nRebuilding appraisals index...")
        reindex_appraisals(env)


if __name__ == "__main__":
    main()
