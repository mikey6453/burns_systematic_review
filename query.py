"""Interactive CLI for asking questions over the ingested PDF corpus.

This is the *grounded* version: every answer is required to come with verbatim
quotes from the retrieved chunks, and each quote is verified against those
chunks before being shown. If the corpus does not actually support the
question, the system refuses rather than fabricating an answer.

Usage:
    py query.py                          # query the PDF corpus
    py query.py --collection appraisals  # query the appraisal judgments
"""
import argparse
import os
import re
import sys

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure") and getattr(_stream, "encoding", "").lower() != "utf-8":
        _stream.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMListwiseRerank
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

# Reuse the verification helpers from appraise.py — single source of truth for
# quote normalization and substring matching.
from appraise import _normalize, locate_quote

# --- Tuning knobs ---
RETRIEVE_K = 10        # fetch a wide net from Qdrant for recall
RERANK_TOP_N = 4       # let the LLM reranker keep the most relevant subset
MAX_UNVERIFIED_RATIO = 0.5  # if more than half of citations don't verify, refuse


# --- Structured output schema (the verification contract) ---

class GroundedAnswer(BaseModel):
    """The LLM is forced to emit this shape so we can audit every claim."""
    refused: bool = Field(
        default=False,
        description=(
            "Set TRUE when the excerpts do not directly address the question. "
            "Refusing is the correct response when evidence is absent — never "
            "guess, stretch, or pattern-match unrelated numbers."
        ),
    )
    refusal_reason: str = Field(
        default="",
        description=(
            "If refused=true, one sentence explaining why the excerpts don't address "
            "the question (e.g., 'corpus contains epidemiology data but no comparison "
            "trials of dressing types in pediatric patients')."
        ),
    )
    answer: str = Field(
        default="",
        description=(
            "2-5 sentence answer using ONLY information present in the excerpts. "
            "Inline-cite each claim with the [filename.pdf p.N] tag from the excerpt "
            "header. Empty string if refused=true."
        ),
    )
    citations: list[str] = Field(
        default_factory=list,
        description=(
            "Verbatim quotes copied EXACTLY from the excerpts. Every numeric value, "
            "percentage, named entity, or specific term in 'answer' MUST be supported "
            "by a quote in this list. Do not paraphrase. Do not insert ellipses. "
            "Do not combine fragments from different excerpts into one quote."
        ),
    )


# --- System prompts (strict, anti-hallucination) ---

_STRICT_RULES = """
STRICT RULES — these prevent fabrication. Violating any of them is a failure:
1. Use ONLY the excerpts. No outside knowledge. No common-sense extrapolation.
2. Every numeric value, percentage, named entity, or specific term in your
   `answer` MUST appear verbatim in one of the quotes in `citations`. If you
   cannot back a claim with an exact quote, do not make the claim.
3. Each citation must be a phrase or sentence copied EXACTLY from an excerpt.
   No paraphrasing. No ellipses. No splicing fragments from different excerpts.
4. If a number appears in an excerpt without a clear label (e.g., a percentage
   in a table cell), do NOT invent a label for it. Quote what the excerpt says
   verbatim and describe it in the words of the excerpt.
5. If the excerpts do NOT directly address the question, set refused=true and
   explain briefly. Refusing is the CORRECT response when evidence is absent.
6. Inline-cite each claim with the [filename.pdf p.N] tag from the excerpt header.
7. Be concise. 2-5 sentences.
""".strip()

SYSTEM_PROMPTS = {
    "papers": (
        "You are a research assistant answering questions about burn-injury "
        "systematic-review papers using ONLY the provided excerpts.\n\n" + _STRICT_RULES
    ),
    "appraisals": (
        "You are answering questions about a corpus of rubric-based evidence "
        "appraisals. Each excerpt is one domain-level judgment for one paper, "
        "including the judgment, agreement across runs, supporting quotes, and "
        "rationale. Do not re-derive ratings — only report what the appraisals say.\n\n"
        + _STRICT_RULES + "\n"
        "8. If asked for a list (e.g., 'which papers had low risk for D2'), "
        "respond as a bulleted list with one bullet per paper."
    ),
}


def get_env():
    load_dotenv()
    return {
        "qdrant_path": os.environ.get("QDRANT_PATH", "./qdrant_data"),
        "papers_collection": os.environ.get("COLLECTION_NAME", "burns_papers"),
        "appraisals_collection": os.environ.get("APPRAISALS_COLLECTION", "burns_appraisals"),
        "embed_model": os.environ.get("EMBED_MODEL", "text-embedding-3-small"),
        "llm_model": os.environ.get("LLM_MODEL", "gpt-4o-mini"),
    }


def format_docs(docs: list[Document]) -> str:
    """Render retrieved chunks with a citation tag the LLM can copy verbatim."""
    return "\n\n---\n\n".join(
        f"[{d.metadata.get('source', '?')} p.{d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )


# --- Verification (the trust mechanism) ---

def verify_citations(citations: list[str], docs: list[Document]) -> tuple[list[dict], list[str]]:
    """Check each cited quote is a substring of one of the retrieved chunks.

    Returns (verified, unverified). Each verified entry has the source filename
    and page taken from the matching chunk's metadata — never from the LLM.
    """
    verified: list[dict] = []
    unverified: list[str] = []
    for q in citations:
        if not q.strip():
            continue
        page, ok = locate_quote(q, docs)
        if not ok:
            unverified.append(q)
            continue
        # Find the source filename of the matching chunk.
        needle = _normalize(q)
        source = "?"
        for d in docs:
            if needle in _normalize(d.page_content):
                source = d.metadata.get("source", "?")
                break
        verified.append({"quote": q, "source": source, "page": page})
    return verified, unverified


_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def numbers_unsupported(answer: str, verified: list[dict]) -> list[str]:
    """Return any numbers in `answer` that don't appear in any verified quote.

    This catches the most common hallucination: the LLM lifts a percentage
    from the retrieved text but pairs it with an invented label. Example
    failure mode: '57% used treatment dressings' when the source says
    '57% of burns were superficial second degree' — the number checks out
    but the label is invented. We can't catch the label invention directly,
    but we can guarantee that every number was at least seen in the evidence.
    """
    answer_nums = set(_NUM_RE.findall(answer))
    if not answer_nums:
        return []
    quote_nums = set()
    for v in verified:
        quote_nums.update(_NUM_RE.findall(v["quote"]))
    return sorted(answer_nums - quote_nums)


# --- Chain ---

def build_chain(env, mode: str):
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set. Copy .env.example to .env and fill it in.")

    collection = env["papers_collection"] if mode == "papers" else env["appraisals_collection"]
    client = QdrantClient(path=env["qdrant_path"])
    if not client.collection_exists(collection):
        hint = "`py ingest.py`" if mode == "papers" else "`py appraise.py --reindex` (after running appraise.py)"
        sys.exit(f"Collection '{collection}' not found. Run {hint} first.")

    embeddings = OpenAIEmbeddings(model=env["embed_model"])
    vector_store = QdrantVectorStore(
        client=client, collection_name=collection, embedding=embeddings
    )
    base_retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVE_K})

    llm = ChatOpenAI(model=env["llm_model"], temperature=0)
    reranker = LLMListwiseRerank.from_llm(llm=llm, top_n=RERANK_TOP_N)
    retriever = ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=base_retriever
    )
    structured_llm = llm.with_structured_output(GroundedAnswer, method="function_calling")

    sys_prompt = SYSTEM_PROMPTS[mode]

    def run(question: str):
        docs = retriever.invoke(question)
        context = format_docs(docs)
        user_prompt = f"Excerpts:\n{context}\n\nQuestion: {question}"
        result: GroundedAnswer = structured_llm.invoke([
            ("system", sys_prompt),
            ("human", user_prompt),
        ])
        verified, unverified = verify_citations(result.citations, docs)
        unsupported = numbers_unsupported(result.answer, verified)
        return result, verified, unverified, unsupported, docs

    return run


# --- Output formatting ---

def _truncate(text: str, n: int = 240) -> str:
    return text if len(text) <= n else text[:n] + "..."


def print_response(result: GroundedAnswer, verified: list[dict],
                   unverified: list[str], unsupported: list[str]) -> None:
    # Hard refusal: the LLM said it can't answer.
    if result.refused:
        print("\n[REFUSED] The corpus does not contain a clear answer.")
        if result.refusal_reason:
            print(f"Reason: {result.refusal_reason}")
        print()
        return

    # Hard refusal: too few citations verified — almost certainly fabricated.
    total_cites = len(result.citations)
    verified_count = len(verified)
    unverified_count = len(unverified)
    if total_cites == 0:
        print("\n[REFUSED] The model produced an answer with NO citations.")
        print("This is treated as ungrounded. Draft answer (rejected):")
        print(f"  {_truncate(result.answer)}\n")
        return
    if total_cites > 0 and verified_count == 0:
        print("\n[REFUSED] The model cited sources, but NONE of the quotes appear "
              "verbatim in the retrieved chunks.")
        print("This is the typical fingerprint of fabrication.")
        print(f"Draft answer (rejected): {_truncate(result.answer)}")
        print(f"Unverified quotes ({unverified_count}):")
        for u in unverified:
            print(f"  [unverified] \"{_truncate(u, 160)}\"")
        print()
        return
    if unverified_count / total_cites > MAX_UNVERIFIED_RATIO:
        print(f"\n[REFUSED] {unverified_count}/{total_cites} citations failed verification "
              f"(threshold: <={int(MAX_UNVERIFIED_RATIO*100)}% may fail). "
              "Treating the answer as ungrounded.")
        print(f"Draft answer (rejected): {_truncate(result.answer)}\n")
        return

    # Acceptable answer — print it with the audit trail.
    print(f"\nAnswer:\n{result.answer}\n")
    print(f"Verified evidence ({verified_count}/{total_cites}):")
    for v in verified:
        print(f"  [ok] \"{_truncate(v['quote'], 200)}\" — {v['source']} p.{v['page']}")
    if unverified_count:
        print(f"\nDropped (unverified — not found verbatim in retrieved chunks):")
        for u in unverified:
            print(f"  [drop] \"{_truncate(u, 200)}\"")
    if unsupported:
        # Numbers in the answer that don't appear in any verified quote.
        # The LLM may have invented or misread these — surface them loudly.
        print(f"\n[warn] {len(unsupported)} number(s) in the answer have no matching "
              f"quote in the verified evidence: {', '.join(unsupported)}")
        print("  Treat these as suspect — re-check against the source PDF before citing.")
    print()


def main():
    parser = argparse.ArgumentParser(description="Ask grounded questions over the corpus or its appraisals.")
    parser.add_argument("--collection", choices=["papers", "appraisals"], default="papers",
                        help="Which index to query: PDF chunks (papers) or rubric judgments (appraisals).")
    args = parser.parse_args()

    env = get_env()
    run = build_chain(env, args.collection)

    label = "papers" if args.collection == "papers" else "appraisals"
    print(f"Burns RAG ({label}) — grounded mode. Ask a question (Ctrl-C to exit).")
    print("Answers without verifiable quotes will be refused.\n")
    while True:
        try:
            question = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not question:
            continue
        try:
            result, verified, unverified, unsupported, _docs = run(question)
        except Exception as e:
            print(f"Error: {e}\n")
            continue
        print_response(result, verified, unverified, unsupported)


if __name__ == "__main__":
    main()
