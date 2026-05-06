"""Shared Qdrant client + embeddings + vector store for the Streamlit pages.

Qdrant local file mode is single-writer: only one QdrantClient can hold the
storage lock at a time. Each Streamlit page caching its own client led to lock
contention when users navigated between pages. By centralizing here, every
page reuses the same instances (st.cache_resource is global per function, so
sharing the function means sharing the resource).
"""
import os

import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


@st.cache_resource(show_spinner="Connecting to Qdrant...")
def get_qdrant_client() -> QdrantClient:
    """One QdrantClient per Streamlit process. Held for the app's lifetime."""
    return QdrantClient(path=os.environ.get("QDRANT_PATH", "./qdrant_data"))


@st.cache_resource(show_spinner="Loading embeddings model...")
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=os.environ.get("EMBED_MODEL", "text-embedding-3-small"))


@st.cache_resource
def get_vector_store(collection: str):
    """Return a QdrantVectorStore for the named collection, or None if it doesn't exist."""
    client = get_qdrant_client()
    if not client.collection_exists(collection):
        return None
    return QdrantVectorStore(
        client=client, collection_name=collection, embedding=get_embeddings()
    )
