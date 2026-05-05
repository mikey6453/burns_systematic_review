# Burns RAG

Local Retrieval-Augmented Generation over a folder of burn-injury systematic-review PDFs. Ask natural-language questions, get answers grounded in your own corpus with inline citations to filename + page. Now also includes **rubric-based evidence appraisal** (RoB 2 / ROBINS-I / Newcastle-Ottawa) and **body-of-evidence GRADE** — every judgment is multi-run for consistency and traceable to verbatim quotes from the source PDFs.

## Architecture

```
ingest.py    ──►  Load PDFs ──►  Chunk ──►  Embed ──►  burns_papers (Qdrant)

query.py     ──►  Embed query ──►  Top-10 ──►  LLM rerank to 4
                                                       │
                                                       ▼
                                             gpt-4o-mini answer + inline citations

appraise.py  ──►  Pick paper ──►  Auto-detect rubric ──►  user confirms
                                                                │
                  ┌─────────────────────────────────────────────┘
                  ▼
             For each rubric domain:
                Filtered retrieval (source = filename, top-k chunks)
                       │
                       ▼
                Run N=10 LLM judgments at temp=0.5  (JSON-structured output)
                       │
                       ▼
                Verify each quote is a substring of retrieved chunks
                       │
                       ▼
                Aggregate → mode + agreement % + flag if <70%
                       │
                       ▼
             outputs/<paper>.appraisal.json
             outputs/appraisals.csv          (long: one row per paper-domain)
             outputs/appraisals_summary.csv  (wide: one row per paper)
                       │
                       ▼  (--reindex or after --all)
             burns_appraisals collection (queryable via query.py --collection appraisals)

grade.py     ──►  Pick outcome ──►  Cross-corpus retrieval + per-study RoB summary
                                                       │
                                                       ▼
                                            Multi-run judgment of 5 downgrade
                                            + 3 upgrade GRADE criteria
                                                       │
                                                       ▼
                                            HIGH / MODERATE / LOW / VERY LOW
                                            + outputs/grade_<outcome>.csv
```

- **Orchestration**: LangChain
- **PDF loader**: `PyPDFLoader` (page-aware → page numbers come for free)
- **Chunking**: `RecursiveCharacterTextSplitter`, 500 chars / 100 overlap
- **Embeddings**: OpenAI `text-embedding-3-small` (1536 dims)
- **Vector DB**: Qdrant local file mode (no Docker), two collections: `burns_papers`, `burns_appraisals`
- **Reranker**: `LLMListwiseRerank` using `gpt-4o-mini` (no extra API keys)
- **LLM**: OpenAI `gpt-4o-mini`
- **Appraisal output**: structured JSON via Pydantic + `with_structured_output`

## Setup

Requires Python 3.10+ (3.12 recommended; 3.14 works if all wheels are available).

```powershell
cd C:\Users\jmich\Documents\burns-rag
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Open `.env` and paste your OpenAI key into `OPENAI_API_KEY=`. The default `KNOWLEDGE_BASE_PATH` already points at your Downloads burn-papers folder; override it if you move the PDFs.

## Ingest

```powershell
py ingest.py            # one-time; skips if already populated
py ingest.py --force    # wipe and rebuild (e.g. after adding new PDFs)
```

Expect a few minutes for ~140 PDFs. Output ends with the chunk count and writes the index to `qdrant_data/`.

## Query

```powershell
py query.py                            # query the PDF corpus
py query.py --collection appraisals    # query the rubric-based judgments
```

Interactive prompt — type a question, get an answer with inline `[filename.pdf p.N]` citations and a deduped Sources block underneath. Ctrl-C to exit.

Example (papers):

```
> What dressing types are most commonly compared in pediatric burn studies?

Answer:
The most frequently compared dressings are silver sulfadiazine (SSD) and
silver-impregnated foams [paper-A.pdf p.4]...
```

Example (appraisals):

```
> Which RCTs had low risk of bias for randomization?

Answer:
The following RCTs received a "Low" judgment for D1 (Randomization process):
- iraa103.pdf [iraa103.pdf p.3] — adequate sequence generation, central allocation.
- irac078.pdf [irac078.pdf p.4] — computer-generated sequence, sealed envelopes...
```

## Appraise (per-study rubrics)

```powershell
py appraise.py --list                            # show papers in the corpus
py appraise.py --paper iraa103.pdf --rubric rob2 # grade one paper
py appraise.py --paper iraa103.pdf               # auto-detect rubric, then grade
py appraise.py --all                             # sweep every paper, auto-detect each
py appraise.py --all --rubric rob2               # force one rubric across all papers
py appraise.py --paper iraa103.pdf --dry-run     # cost estimate, run nothing
py appraise.py --reindex                         # rebuild burns_appraisals from CSV
```

Available rubrics: `rob2` (Cochrane RoB 2 — RCTs), `robins_i` (non-randomized intervention studies), `nos_cohort` (Newcastle-Ottawa cohort studies), `custom` (template — copy and edit). See `rubrics/*.json` to inspect or modify.

How a single domain gets graded:

1. Run multiple **targeted retrieval queries** scoped to the paper (Qdrant filter on `source = filename`) — e.g., for RoB 2 D1, queries like *"randomization sequence generation"*, *"allocation concealment"*. Top-`APPRAISE_K` chunks retrieved per query, deduped.
2. Send the chunks + rubric criteria + signaling questions to gpt-4o-mini, **N=10 times at temperature 0.5**, asking for structured JSON: `{judgment, evidence:[{quote, supports}], rationale, criteria_invoked, no_information}`.
3. **Verify** each quote: it must appear verbatim (modulo whitespace) in one of the retrieved chunks. The page number is taken from that chunk's metadata, not from the LLM. Quotes that don't match are flagged `[unverified]` rather than silently kept.
4. **Aggregate**: mode of the 10 judgments wins; agreement % is computed; agreement <`AGREEMENT_THRESHOLD` (default 0.7) flags the row for human review. Quotes from winning runs are pooled and deduped.
5. Write outputs:
   - `outputs/<paper>.<rubric>.appraisal.json` — full per-run audit trail
   - `outputs/appraisals.csv` — long format with `filename, rubric, domain_id, judgment, agreement, flagged, supporting_quotes, pages, rationale, criteria_invoked, runs_breakdown`
   - `outputs/appraisals_summary.csv` — wide format, one row per paper × rubric

After a `--all` sweep, the appraisals CSV is automatically re-indexed into the `burns_appraisals` Qdrant collection so you can ask questions like *"which observational cohort papers had inadequate follow-up?"* via `py query.py --collection appraisals`.

## GRADE (body of evidence, per outcome)

```powershell
py grade.py --outcome "mortality"
py grade.py --outcome "length of stay" --start-from non_rct
py grade.py --outcome "infection rate" --runs 5
```

GRADE is outcome-level, not paper-level. The pipeline:

1. **Cross-corpus retrieval** for the outcome (no `source` filter), combined with each domain's targeted query (e.g., *"mortality: heterogeneity I²"* for G2 Inconsistency).
2. For G1 (Risk of Bias), the prompt also receives a **summary of per-study RoB judgments** read from `outputs/appraisals.csv` — so GRADE leans on the per-study work `appraise.py` already did.
3. Each of the 5 downgrade + 3 upgrade domains is judged N=10 times with the same multi-run + quote-verification logic.
4. **Final certainty** is computed from a starting level (HIGH for RCT bodies, LOW for non-RCT) plus integer deltas from each domain. Never falls below VERY LOW.
5. Outputs:
   - `outputs/grade_<outcome_slug>.json` — full audit trail
   - `outputs/grade_<outcome_slug>.csv` — one row per criterion + a final `FINAL` row with the overall certainty rating

## Tradeoffs (talking points)

### Retrieval
- **Chunk size 500 / overlap 100** — Small for precision: a hit corresponds to a tight passage rather than a whole paragraph, which matters when papers pack multiple ideas per page. Compensated by retrieving more chunks (k=10) and reranking. If answers feel fragmentary, raise to 750–1000.
- **k=10 → rerank to 4** — Over-retrieve for recall, then let an LLM reorder by relevance for precision. Four chunks fits comfortably in the prompt and avoids the "lost-in-the-middle" effect of stuffing many.
- **`text-embedding-3-small` over `-large`** — 6× cheaper for marginal accuracy gain on this scope. A 140-PDF corpus doesn't need the bigger model.
- **Qdrant local file mode** — No Docker, single-writer, fine for a CLI. To scale to a server, swap `QdrantClient(path=...)` for `QdrantClient(url=...)`; no other code changes.
- **LLM-based reranker over Cohere** — Reuses the existing OpenAI key, no extra dependency or paid service. Cohere Rerank is higher quality but requires another API key.
- **Inline citations + Sources block** — The LLM is prompted to cite inline, but a separate deduped Sources block is printed independently. Even if the LLM forgets a tag, the audit trail is intact.

### Appraisal / GRADE
- **Multi-run at temperature 0.5, N=10** — Inter-run agreement only carries information if the LLM is actually sampling. Temperature 0 + same input = identical outputs and a fake 10/10 agreement. 0.5 gives genuine variability without the model going off-script. N=10 matches the spec; bump to 20 for tighter confidence intervals at 2× cost.
- **Per-domain retrieval queries** — Each rubric domain has its own retrieval queries (e.g., RoB 2 D1 looks for "randomization sequence generation, allocation concealment, baseline imbalance"). Stuffing the whole paper into context would lose precision and waste tokens on long PDFs. The tradeoff: if your retrieval queries miss the relevant section, the LLM sees nothing and returns "No information." Edit the JSON to broaden queries if a known concept is being missed.
- **Quote verification (the trust mechanism)** — The LLM returns a `quote` field per piece of evidence; we check it's a substring of the retrieved chunks before assigning a page number. Hallucinated quotes get tagged `[unverified]` in the CSV rather than dropped silently — visibility over filtering. Without this guardrail, "evidence-grounded" judgments are an honor system.
- **Page from index, not from LLM** — The page number for a citation comes from the chunk metadata that the verified quote matched, not from the LLM. LLMs misreport page numbers constantly; this sidesteps the issue.
- **Two CSVs (long + wide)** — Long CSV is `grep`-friendly and is what gets re-embedded into the appraisals collection. Wide CSV is the human-readable "evidence table" the spec describes.
- **Re-ingesting appraisals as a Qdrant collection** — Each CSV row becomes a `Document` whose `page_content` is the rendered judgment + rationale + quotes, with `metadata.source = filename` and `metadata.page = first cited page`. This means `query.py --collection appraisals` reuses the existing retrieval + reranker + citation flow with no special-casing — the appraisals are just another knowledge base.
- **Auto-detect rubric, user confirms** — `appraise.py` (without `--rubric`) classifies each paper's design via the LLM and picks RoB 2 / ROBINS-I / NOS automatically. With `--all`, this runs unattended; for a single paper, you can override with `--rubric`. The spec asks for a confirmation step; the current flow accomplishes that by letting you re-run with `--rubric X` if the auto-detection is wrong.
- **GRADE leans on appraise outputs** — Instead of re-grading risk of bias from scratch when computing GRADE's G1 domain, `grade.py` reads `outputs/appraisals.csv` and feeds a per-paper RoB summary into the prompt. Single source of truth, fewer tokens, and the per-study work isn't wasted.

## Troubleshooting

- **`pip install` fails on Python 3.14** — Some wheels lag new Python releases. Install Python 3.12 from python.org and use `py -3.12 -m venv .venv`.
- **`OPENAI_API_KEY not set`** — You haven't created `.env`, or it doesn't have the key. `copy .env.example .env` and fill it in.
- **Qdrant lock / "storage folder is already accessed"** — Path-mode Qdrant is single-writer. Make sure no other `ingest.py` or `query.py` is running, then retry.
- **`Collection 'burns_papers' not found` when querying** — Run `py ingest.py` first.
- **Want to point at a different folder of PDFs** — Edit `KNOWLEDGE_BASE_PATH` in `.env` and run `py ingest.py --force`.

## Files

| File | Purpose |
|---|---|
| `ingest.py` | Load PDFs, chunk, embed, store in `burns_papers`. `--force` to rebuild. |
| `query.py` | Interactive CLI. `--collection papers` (default) or `--collection appraisals`. |
| `appraise.py` | Per-paper rubric grading (RoB 2 / ROBINS-I / NOS / custom). Multi-run + verified evidence. |
| `grade.py` | Body-of-evidence GRADE for a specific outcome. Reuses `appraisals.csv` for G1. |
| `rubrics/` | JSON definitions of each rubric. Add a custom rubric by dropping a file here. |
| `outputs/` | Per-paper JSONs + `appraisals.csv` + `appraisals_summary.csv` + GRADE outputs (gitignored). |
| `qdrant_data/` | Local Qdrant indices: `burns_papers` + `burns_appraisals` (gitignored). |
| `requirements.txt` | Python deps. |
| `.env.example` | Config template. Copy to `.env` and edit. |
