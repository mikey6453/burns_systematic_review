"""Ingest PDFs from KNOWLEDGE_BASE_PATH into a local Qdrant collection.

Usage:
    py ingest.py            # skips if collection already populated
    py ingest.py --force    # wipes the collection and re-ingests
"""
import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# --- Tuning knobs (see README for tradeoff notes) ---
CHUNK_SIZE = 500       # chars (~125 tokens) — small for precise retrieval
CHUNK_OVERLAP = 100    # chars — preserve meaning across chunk boundaries
EMBED_DIM = 1536       # text-embedding-3-small output size


def get_env():
    load_dotenv()
    return {
        "kb_path": os.environ["KNOWLEDGE_BASE_PATH"],
        "qdrant_path": os.environ.get("QDRANT_PATH", "./qdrant_data"),
        "collection": os.environ.get("COLLECTION_NAME", "burns_papers"),
        "embed_model": os.environ.get("EMBED_MODEL", "text-embedding-3-small"),
    }


def collection_is_populated(client: QdrantClient, name: str) -> bool:
    if not client.collection_exists(name):
        return False
    return client.count(collection_name=name, exact=True).count > 0


def load_pdfs(kb_path: str):
    """Yield (filename, page_documents) for every PDF in kb_path."""
    pdf_paths = sorted(Path(kb_path).glob("*.pdf"))
    if not pdf_paths:
        sys.exit(f"No PDFs found in {kb_path}")
    print(f"Found {len(pdf_paths)} PDFs in {kb_path}")
    for i, path in enumerate(pdf_paths, 1):
        try:
            pages = PyPDFLoader(str(path)).load()
        except Exception as e:
            print(f"  [{i}/{len(pdf_paths)}] SKIP {path.name}: {e}")
            continue
        # PyPDFLoader sets metadata.source to the full path and metadata.page (0-indexed).
        # We overwrite source to the bare filename (cleaner citations) and bump page to 1-indexed.
        for p in pages:
            p.metadata["source"] = path.name
            p.metadata["page"] = p.metadata.get("page", 0) + 1
        print(f"  [{i}/{len(pdf_paths)}] {path.name} ({len(pages)} pages)")
        yield pages


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into Qdrant.")
    parser.add_argument("--force", action="store_true", help="Delete existing collection and re-ingest")
    args = parser.parse_args()

    env = get_env()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set. Copy .env.example to .env and fill it in.")

    t0 = time.time()
    client = QdrantClient(path=env["qdrant_path"])

    if collection_is_populated(client, env["collection"]):
        if not args.force:
            count = client.count(collection_name=env["collection"], exact=True).count
            print(f"Collection '{env['collection']}' already has {count} vectors. "
                  "Skipping. Use --force to rebuild.")
            return
        print(f"--force: deleting existing collection '{env['collection']}'")
        client.delete_collection(env["collection"])

    client.create_collection(
        collection_name=env["collection"],
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    embeddings = OpenAIEmbeddings(model=env["embed_model"])
    vector_store = QdrantVectorStore(
        client=client, collection_name=env["collection"], embedding=embeddings
    )

    total_pages = 0
    total_chunks = 0
    for pages in load_pdfs(env["kb_path"]):
        total_pages += len(pages)
        chunks = splitter.split_documents(pages)
        if chunks:
            vector_store.add_documents(chunks)
            total_chunks += len(chunks)

    elapsed = time.time() - t0
    print(f"\nIngest complete: {total_pages} pages -> {total_chunks} chunks "
          f"in {elapsed:.1f}s")
    print(f"Qdrant data persisted at: {env['qdrant_path']}")


if __name__ == "__main__":
    main()
