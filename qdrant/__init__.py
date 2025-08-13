"""
Qdrant Vector Database Module for RFP RAG System
Provides client, indexer, and retriever functionality
"""

# Import functions, not classes (since we don't have classes)
from .client import get_qdrant_client, setup_collections_dynamic, INTERNAL_COLLECTION, RFP_QA_COLLECTION
from .indexer import upsert_documents, index_internal_documents, index_rfp_qa_pairs

__all__ = [
    'get_qdrant_client', 
    'setup_collections_dynamic', 
    'INTERNAL_COLLECTION', 
    'RFP_QA_COLLECTION',
    'upsert_documents', 
    'index_internal_documents', 
    'index_rfp_qa_pairs'
]
