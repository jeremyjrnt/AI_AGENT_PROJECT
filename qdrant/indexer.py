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
import os
from pathlib import Path
import pandas as pd
from datetime import datetime

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


def index_internal_data_with_source(data_folder_path: str, source_name: str = "internal_docs") -> int:
    """
    Index internal documentation data into vector database with internal source metadata.
    
    Args:
        data_folder_path: Path to folder containing internal documents
        source_name: Name to identify the source (default: "internal_docs")
        
    Returns:
        int: Number of documents indexed
        
    Example:
        count = index_internal_data_with_source("/path/to/internal/docs", "company_handbook")
    """
    data_path = Path(data_folder_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data folder not found: {data_folder_path}")
    
    documents = []
    
    # Process text files (.txt, .md)
    for file_path in data_path.rglob("*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    documents.append({
                        "text": content,
                        "metadata": {
                            "source": "internal",
                            "source_type": "internal_docs",
                            "source_name": source_name,
                            "file_path": str(file_path),
                            "file_name": file_path.name,
                            "indexed_at": datetime.now().isoformat(),
                            "document_type": "internal_documentation"
                        }
                    })
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Process markdown files
    for file_path in data_path.rglob("*.md"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    documents.append({
                        "text": content,
                        "metadata": {
                            "source": "internal",
                            "source_type": "internal_docs",
                            "source_name": source_name,
                            "file_path": str(file_path),
                            "file_name": file_path.name,
                            "indexed_at": datetime.now().isoformat(),
                            "document_type": "internal_documentation"
                        }
                    })
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    if documents:
        return index_internal_documents(documents)
    else:
        print(f"No documents found in {data_folder_path}")
        return 0


def index_completed_rfp(rfp_excel_path: str, rfp_source_info: Dict[str, Any] = None) -> int:
    """
    Index completed RFP Q&A pairs into vector database.
    The vector is created from the question text, and the answer + metadata are stored as payload.
    
    Args:
        rfp_excel_path: Path to completed RFP Excel file
        rfp_source_info: Additional source information (client, project, etc.)
        
    Returns:
        int: Number of Q&A pairs indexed
        
    Example:
        source_info = {
            "client_name": "TechCorp",
            "project": "Cloud Migration RFP",
            "completion_date": "2025-08-18"
        }
        count = index_completed_rfp("/path/to/completed_rfp.xlsx", source_info)
    """
    excel_path = Path(rfp_excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"RFP Excel file not found: {rfp_excel_path}")
    
    # Default source info
    if rfp_source_info is None:
        rfp_source_info = {}
    
    try:
        # Read Excel file
        df = pd.read_excel(excel_path)
        
        # Validate required columns (flexible column names)
        question_col = None
        answer_col = None
        comment_col = None
        
        for col in df.columns:
            col_lower = col.lower().strip()
            if 'question' in col_lower:
                question_col = col
            elif 'answer' in col_lower:
                answer_col = col
            elif 'comment' in col_lower or 'note' in col_lower:
                comment_col = col
        
        if not question_col or not answer_col:
            raise ValueError("Excel must contain 'Question' and 'Answer' columns")
        
        qa_documents = []
        
        for idx, row in df.iterrows():
            question = str(row[question_col]).strip()
            answer = str(row[answer_col]).strip()
            comment = str(row[comment_col]).strip() if comment_col else ""
            
            # Skip empty questions
            if not question or question.lower() in ['nan', 'none', '']:
                continue
            
            # Create metadata
            metadata = {
                "source": "past-rfp",
                "source_type": "completed_rfp",
                "question": question,
                "answer": answer,
                "comment": comment,
                "rfp_file": excel_path.name,
                "rfp_path": str(excel_path),
                "question_index": idx,
                "indexed_at": datetime.now().isoformat(),
                "document_type": "rfp_qa_pair"
            }
            
            # Add custom source info
            metadata.update(rfp_source_info)
            
            qa_documents.append({
                "text": question,  # The vector will be created from the question
                "metadata": metadata
            })
        
        if qa_documents:
            print(f"Indexing {len(qa_documents)} Q&A pairs from {excel_path.name}")
            return index_rfp_qa_pairs(qa_documents)
        else:
            print(f"No valid Q&A pairs found in {excel_path}")
            return 0
            
    except Exception as e:
        print(f"Error processing RFP Excel file {rfp_excel_path}: {e}")
        raise


def batch_index_completed_rfps(completed_rfps_folder: str) -> int:
    """
    Batch index all completed RFP Excel files from a folder.
    
    Args:
        completed_rfps_folder: Path to folder containing completed RFP Excel files
        
    Returns:
        int: Total number of Q&A pairs indexed
        
    Example:
        total = batch_index_completed_rfps("/path/to/completed_RFPs/")
    """
    folder_path = Path(completed_rfps_folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Completed RFPs folder not found: {completed_rfps_folder}")
    
    total_indexed = 0
    processed_files = []
    
    # Process all Excel files in the folder
    for excel_file in folder_path.glob("*.xlsx"):
        try:
            # Extract source info from filename if possible
            source_info = {
                "batch_indexed": True,
                "batch_date": datetime.now().isoformat(),
                "source_folder": str(folder_path)
            }
            
            # Try to extract info from filename (e.g., "client_project_date.xlsx")
            filename_parts = excel_file.stem.split('_')
            if len(filename_parts) >= 2:
                source_info["derived_client"] = filename_parts[0]
                source_info["derived_project"] = '_'.join(filename_parts[1:-1]) if len(filename_parts) > 2 else filename_parts[1]
            
            count = index_completed_rfp(str(excel_file), source_info)
            total_indexed += count
            processed_files.append(excel_file.name)
            print(f"✅ Indexed {count} Q&A pairs from {excel_file.name}")
            
        except Exception as e:
            print(f"❌ Error processing {excel_file.name}: {e}")
    
    print(f"\n🎉 Batch indexing complete!")
    print(f"📁 Processed {len(processed_files)} files")
    print(f"📊 Total Q&A pairs indexed: {total_indexed}")
    
    return total_indexed
