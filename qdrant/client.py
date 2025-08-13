#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qdrant Vector Database Client
- Dynamic embedding provider (HuggingFace now, OpenAI later)
- Idempotent collection setup with automatic vector size detection
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import settings

# Collections
INTERNAL_COLLECTION = "internal_knowledge_base"  # stable
RFP_QA_COLLECTION = "rfp_qa_history"             # evolving


# ---------- Embeddings utilities (LangChain) ----------
def _get_huggingface_embeddings():
    """Lazy import to avoid heavy load at import time."""
    # pip install langchain-huggingface sentence-transformers
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL, encode_kwargs={"normalize_embeddings": True})


def _get_openai_embeddings():
    """Lazy import to avoid requiring OpenAI until you switch."""
    # pip install langchain-openai
    from langchain_openai import OpenAIEmbeddings
    model = getattr(settings, "OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    return OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY, model=model)


def get_embeddings():
    """
    Return a LangChain embeddings object according to EMBEDDING_PROVIDER.
    Supported: "huggingface" (free), "openai".
    """
    provider = getattr(settings, "EMBEDDING_PROVIDER", "huggingface").lower()
    if provider == "openai":
        if not getattr(settings, "OPENAI_API_KEY", ""):
            raise ValueError("OPENAI_API_KEY is missing in settings for provider='openai'.")
        return _get_openai_embeddings()
    # default -> huggingface
    return _get_huggingface_embeddings()


def detect_vector_size(embeddings_obj) -> int:
    """
    Determine embedding vector dimension by encoding a tiny probe.
    We never hardcode sizes; we ask the model once.
    """
    probe: List[str] = ["__dim_probe__"]
    vecs = embeddings_obj.embed_documents(probe)  # -> List[List[float]]
    if not vecs or not isinstance(vecs[0], list) or len(vecs[0]) == 0:
        raise RuntimeError("Failed to detect embedding dimension from provider.")
    return len(vecs[0])


# ---------- Qdrant utilities ----------
def get_qdrant_client() -> QdrantClient:
    """Return a configured Qdrant client (Cloud or local)."""
    kwargs = {"url": settings.QDRANT_URL, "timeout": 30}
    if getattr(settings, "QDRANT_API_KEY", None):
        kwargs["api_key"] = settings.QDRANT_API_KEY
    return QdrantClient(**kwargs)


def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    """Check if a collection exists on the server."""
    cols = client.get_collections().collections
    return any(c.name == collection_name for c in cols)


def _current_vector_size(client: QdrantClient, collection_name: str) -> Optional[int]:
    """
    Return current vector size if single-vector schema, else None.
    """
    info = client.get_collection(collection_name)
    vectors = info.config.params.vectors
    if isinstance(vectors, VectorParams):
        return vectors.size
    return None  # named-vectors schema not handled here


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    expected_size: int,
    distance: Distance = Distance.COSINE,
    recreate_if_mismatch: bool = False,
) -> None:
    """
    Ensure the collection exists with the expected vector size.
    - If missing: create it.
    - If size mismatch:
        * recreate if `recreate_if_mismatch=True`
        * else raise ValueError to avoid accidental data loss.
    """
    if not collection_exists(client, collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=expected_size, distance=distance),
        )
        return

    current = _current_vector_size(client, collection_name)
    if current is None:
        raise ValueError(
            f"Collection '{collection_name}' uses a named-vector schema; "
            f"adapt ensure_collection() before using this helper."
        )
    if current != expected_size:
        if recreate_if_mismatch:
            client.delete_collection(collection_name=collection_name)
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=expected_size, distance=distance),
            )
        else:
            raise ValueError(
                f"Vector size mismatch for '{collection_name}': existing={current}, expected={expected_size}. "
                f"Pass recreate_if_mismatch=True to recreate (will delete data)."
            )


def setup_collections_dynamic(reset_rfp_history: bool = False) -> int:
    """
    Detect embedding dimension from current provider/model,
    then ensure both collections are compatible.
    - INTERNAL_COLLECTION: never recreated silently
    - RFP_QA_COLLECTION : optional reset if requested
    Returns the detected vector size.
    """
    # 1) Get embeddings & detect size
    emb = get_embeddings()
    vector_size = detect_vector_size(emb)

    # 2) Ensure collections
    client = get_qdrant_client()
    ensure_collection(client, INTERNAL_COLLECTION, vector_size, Distance.COSINE, recreate_if_mismatch=False)
    ensure_collection(client, RFP_QA_COLLECTION, vector_size, Distance.COSINE, recreate_if_mismatch=reset_rfp_history)
    return vector_size


def get_collection_info(client: QdrantClient, collection_name: str) -> Optional[Dict[str, Any]]:
    """Return lightweight collection info (or None if not found)."""
    if not collection_exists(client, collection_name):
        return None
    info = client.get_collection(collection_name)
    vectors = info.config.params.vectors
    size = vectors.size if isinstance(vectors, VectorParams) else None
    return {
        "name": collection_name,
        "vector_size": size,
        "distance": vectors.distance.value if isinstance(vectors, VectorParams) else "named-vectors",
        "points_count": info.points_count,
        "status": info.status.value,
    }


def get_all_collections_info() -> Dict[str, Any]:
    """Convenience helper to inspect connection and both collections."""
    client = get_qdrant_client()
    return {
        "connection": {
            "qdrant_url": settings.QDRANT_URL,
            "has_api_key": bool(getattr(settings, "QDRANT_API_KEY", "")),
            "provider": getattr(settings, "EMBEDDING_PROVIDER", "huggingface"),
            "model": getattr(settings, "EMBEDDING_MODEL", ""),
            "openai_model": getattr(settings, "OPENAI_EMBEDDING_MODEL", ""),
        },
        "collections": {
            "internal": get_collection_info(client, INTERNAL_COLLECTION),
            "rfp_qa": get_collection_info(client, RFP_QA_COLLECTION),
        },
    }
