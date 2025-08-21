#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qdrant Vector Database Client
- Dynamic embedding provider (HuggingFace now, OpenAI later)
- Idempotent collection setup with automatic vector size detection
- Usage tracking for cost monitoring
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import settings

# Usage tracking
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
try:
    from tokens_count.usage_tracker import LocalUsageTracker
    usage_tracker = LocalUsageTracker("embeddings_usage.json")
except ImportError:
    usage_tracker = None

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
    from langchain_openai import AzureOpenAIEmbeddings
    
    # Utilise directement Azure OpenAI
    if settings.is_azure_openai_configured():
        return AzureOpenAIEmbeddings(
            api_key=settings.AZURE_OPENAI_API_KEY,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            deployment=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        )
    else:
        raise ValueError("Azure OpenAI is not configured. Please check your .env file.")


def get_embeddings():
    """
    Return a LangChain embeddings object according to EMBEDDING_PROVIDER.
    Supported: "huggingface" (free), "openai" (Azure OpenAI only).
    Includes usage tracking for cost monitoring.
    """
    provider = getattr(settings, "EMBEDDING_PROVIDER", "huggingface").lower()
    if provider == "openai":
        if not settings.is_azure_openai_configured():
            raise ValueError("Azure OpenAI is not configured for provider='openai'. Check your .env file.")
        
        # Wrap Azure OpenAI embeddings with usage tracking
        base_embeddings = _get_openai_embeddings()
        if usage_tracker:
            return TrackedEmbeddings(base_embeddings, "azure_openai", "team11-embedding")
        return base_embeddings
    
    # default -> huggingface  
    base_embeddings = _get_huggingface_embeddings()
    if usage_tracker:
        return TrackedEmbeddings(base_embeddings, "huggingface", settings.EMBEDDING_MODEL)
    return base_embeddings


class TrackedEmbeddings:
    """Track embedding usage with memory-only storage"""
    
    def __init__(self, base_embeddings, provider, model):
        self.base_embeddings = base_embeddings
        self.provider = provider
        self.model = model
    
    def embed_documents(self, texts):
        """Embed documents avec tracking"""
        # Log l'utilisation avant l'appel
        if usage_tracker:
            usage_tracker.log_embedding_request(texts, self.provider, self.model)
        
        # Faire l'embedding réel
        return self.base_embeddings.embed_documents(texts)
    
    def embed_query(self, text):
        """Embed query avec tracking"""
        # Log l'utilisation avant l'appel  
        if usage_tracker:
            usage_tracker.log_embedding_request([text], self.provider, self.model)
        
        # Faire l'embedding réel
        return self.base_embeddings.embed_query(text)
    
    def __getattr__(self, name):
        """Déléguer tous les autres attributs à l'embedding de base"""
        return getattr(self.base_embeddings, name)


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
            "azure_configured": settings.is_azure_openai_configured(),
            "azure_deployment": settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT if settings.is_azure_openai_configured() else None,
        },
        "collections": {
            "internal": get_collection_info(client, INTERNAL_COLLECTION),
            "rfp_qa": get_collection_info(client, RFP_QA_COLLECTION),
        },
    }
