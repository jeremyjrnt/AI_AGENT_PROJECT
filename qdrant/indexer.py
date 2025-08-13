#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qdrant Indexer (minimal)
- Input: list of {"text": str, "metadata": dict} items
- Embeds texts with current provider and upserts to a target collection
"""

from __future__ import annotations
from typing import List, Dict, Any
from uuid import uuid4

from qdrant_client.http.models import PointStruct
from .client import (
    get_qdrant_client,
    get_embeddings,
    INTERNAL_COLLECTION,
    RFP_QA_COLLECTION,
)

JsonDoc = Dict[str, Any]

def upsert_documents(docs: List[JsonDoc], collection_name: str) -> int:
    """
    Minimal upsert: embed texts and attach metadata as payload.
    Assumes the collection already exists and has the correct vector size.
    
    Args:
        docs: List of documents with 'text' (str) and optional 'metadata' (dict)
        collection_name: Target collection name
        
    Returns:
        int: Number of documents successfully inserted
        
    Example:
        docs = [
            {"text": "Hello world", "metadata": {"source": "test.html"}},
            {"text": "Another doc", "metadata": {"category": "qa"}}
        ]
        count = upsert_documents(docs, "my_collection")
    """
    if not docs:
        return 0

    # 1) Extract texts (validate input)
    texts = []
    payloads = []
    for i, d in enumerate(docs):
        txt = d.get("text")
        if not isinstance(txt, str) or not txt.strip():
            raise ValueError(f"Doc[{i}] must include non-empty 'text' (str).")
        texts.append(txt)
        md = d.get("metadata") or {}
        if not isinstance(md, dict):
            raise ValueError(f"Doc[{i}] 'metadata' must be a dict if provided.")
        # Optionally store original text for debugging/retrieval
        md.setdefault("text", txt)
        payloads.append(md)

    # 2) Get embeddings (lazy-loaded provider)
    emb = get_embeddings()
    vectors = emb.embed_documents(texts)
    if len(vectors) != len(texts):
        raise RuntimeError("Embedding count mismatch.")

    # 3) Build points
    points = [
        PointStruct(id=str(uuid4()), vector=vec, payload=pld)
        for vec, pld in zip(vectors, payloads)
    ]

    # 4) Upsert
    client = get_qdrant_client()
    client.upsert(collection_name=collection_name, points=points)
    return len(points)

# Convenience wrappers
def index_internal_documents(docs: List[JsonDoc]) -> int:
    """Upsert into the stable internal collection."""
    return upsert_documents(docs, INTERNAL_COLLECTION)


def index_rfp_qa_pairs(qa_pairs: List[JsonDoc]) -> int:
    """Upsert into the evolving RFP Q&A history collection."""
    return upsert_documents(qa_pairs, RFP_QA_COLLECTION)
