#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qdrant Indexer (minimal)
- Two main functions: upsert_data and upsert_rfp
- All embedding logic integrated directly
"""

from __future__ import annotations
from typing import Union
from uuid import uuid4
import sys
from pathlib import Path
from datetime import datetime

from qdrant_client.http.models import PointStruct
from .client import (
    get_qdrant_client,
    get_embeddings,
    INTERNAL_COLLECTION,
    RFP_QA_COLLECTION,
)
from .rfp_tracker import get_rfp_tracker

# Import parsers
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from parsers.internal_parser import parse_folder_to_data
from parsers.rfp_parser import RFPParser

def upsert_data(
    folder_path: Union[str, Path], 
    target_chars: int = 1500, 
    overlap: int = 300, 
    mode: str = "hybrid",
    collection_name: str = INTERNAL_COLLECTION
) -> int:
    """
    Parse internal data using internal_parser and upsert to Qdrant.
    
    Args:
        folder_path: Path to folder containing documents to parse
        target_chars: Target characters per chunk
        overlap: Overlap between chunks
        mode: Parsing mode ('dev', 'prod', 'hybrid')
        collection_name: Target Qdrant collection
        
    Returns:
        int: Number of documents successfully indexed
    """
    print(f"🔄 Parsing data from: {folder_path}")
    print(f"   Mode: {mode}, Target chars: {target_chars}, Overlap: {overlap}")
    
    # Parse using internal_parser
    parsed_docs = parse_folder_to_data(
        folder_path=folder_path,
        target_chars=target_chars,
        overlap=overlap,
        mode=mode
    )
    
    if not parsed_docs:
        print("❌ No documents parsed")
        return 0
    
    print(f"✅ Parsed {len(parsed_docs)} documents")
    
    # Extract texts for embedding (prioritize enhanced_text)
    texts = []
    payloads = []
    
    for i, doc in enumerate(parsed_docs):
        # Check different locations for enhanced_text
        enhanced_txt = ""
        
        # Try different possible locations for enhanced_text
        if doc.get("enhanced_text"):
            enhanced_txt = doc.get("enhanced_text")
        elif doc.get("llm", {}).get("enhanced_text"):
            enhanced_txt = doc.get("llm", {}).get("enhanced_text")
        elif doc.get("metadata", {}).get("llm", {}).get("enhanced_text"):
            enhanced_txt = doc.get("metadata", {}).get("llm", {}).get("enhanced_text")
        
        original_txt = doc.get("text", "")
        
        # Use enhanced_text if available and non-empty, otherwise fallback to text
        if enhanced_txt and enhanced_txt.strip():
            embedding_text = enhanced_txt
            embedded_field = "enhanced_text"
            print(f"📝 Using enhanced_text for embedding (doc {i+1})")
        elif original_txt and original_txt.strip():
            embedding_text = original_txt
            embedded_field = "text"
            print(f"📝 Using text for embedding (doc {i+1}) - no enhanced_text available")
        else:
            print(f"⚠️  Skipping document {i+1}: no valid text for embedding")
            continue
        
        texts.append(embedding_text)
        
        # Prepare metadata
        metadata = doc.get("metadata", {})
        metadata.update({
            "parsing_mode": mode,
            "target_chars": target_chars,
            "overlap": overlap,
            "indexed_at": datetime.now().isoformat(),
            "document_type": "internal_data",
            "embedded_field": embedded_field
        })
        
        # Store both text fields if available, but avoid duplication
        if original_txt:
            metadata["text"] = original_txt
        
        # Only store enhanced_text at root level if it's not already in metadata structure
        if enhanced_txt and embedded_field == "enhanced_text":
            # Check if enhanced_text is already stored in metadata.llm structure
            if not (metadata.get("llm", {}).get("enhanced_text")):
                metadata["enhanced_text"] = enhanced_txt
        
        if doc.get('summary'):
            metadata["summary"] = doc.get('summary')
        
        payloads.append(metadata)
    
    if not texts:
        print("❌ No valid texts for embedding")
        return 0
    
    print(f"🗃️ Indexing {len(texts)} documents to {collection_name}...")
    
    # Get embeddings
    emb = get_embeddings()
    vectors = emb.embed_documents(texts)
    if len(vectors) != len(texts):
        raise RuntimeError("Embedding count mismatch.")
    
    # Build points
    points = [
        PointStruct(id=str(uuid4()), vector=vec, payload=pld)
        for vec, pld in zip(vectors, payloads)
    ]
    
    # Upsert to Qdrant
    client = get_qdrant_client()
    client.upsert(collection_name=collection_name, points=points)
    return len(points)


def upsert_rfp(
    rfp_file_path: Union[str, Path],
    question_vectors: list,
    submitter_name: str,
    source: str = None,
    collection_name: str = RFP_QA_COLLECTION,
    auto_cleanup: bool = True
) -> int:
    """
    Parse RFP Excel file with 'Question', 'Answer', 'Comments' columns and upsert to Qdrant.
    
    Args:
        rfp_file_path: Path to RFP Excel file with Question/Answer/Comments columns
        question_vectors: Pre-computed embedding vectors for questions (list of vectors)
        submitter_name: Single submitter name for the entire RFP
        source: Source or origin of this RFP (e.g., Client, Internal, Tender)
        collection_name: Target Qdrant collection
        auto_cleanup: Whether to perform automatic cleanup of old RFPs
        
    Returns:
        int: Number of Q&A pairs successfully indexed
    """
    import pandas as pd
    
    rfp_path = Path(rfp_file_path)
    if not rfp_path.exists():
        raise FileNotFoundError(f"RFP file not found: {rfp_file_path}")
    
    print(f"🔄 Parsing RFP from: {rfp_path.name}")
    print(f"📝 Submitter: {submitter_name}")
    print(f"🏢 Source: {source or 'Not specified'}")
    print(f"🔢 Received {len(question_vectors)} pre-computed vectors")
    
    # Get RFP tracker and assign next RFP number (age counter)
    tracker = get_rfp_tracker()
    rfp_age = tracker.get_next_rfp_number()
    
    try:
        # Read Excel file
        df = pd.read_excel(rfp_path)
        print(f"📋 Loaded Excel with {len(df)} rows and columns: {list(df.columns)}")
        
        # Validate required columns
        required_cols = ['Question', 'Answer', 'Comments']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}. Found: {list(df.columns)}")
        
        # Extract question data
        questions = []
        payloads = []
        valid_vectors = []
        
        for index, row in df.iterrows():
            question = str(row['Question']).strip()
            answer = str(row['Answer']).strip() 
            comments = str(row['Comments']).strip()
            
            # Skip empty questions
            if not question or question.lower() in ['nan', 'none', '']:
                print(f"⚠️  Skipping row {index+1}: empty question")
                continue
            
            # Check if we have corresponding vector
            if index >= len(question_vectors):
                print(f"⚠️  Skipping row {index+1}: no corresponding vector")
                continue
            
            # Get validator name (try different column names)
            validator_name = ""
            for val_col in ['Validator Name', 'Validator', 'Validator_Name', 'ValidatorName', 'validator_name']:
                if val_col in df.columns:
                    validator_name = str(row.get(val_col, '')).strip()
                    break
            
            questions.append(question)
            valid_vectors.append(question_vectors[index])
            
            # Prepare enhanced metadata
            metadata = {
                # Core RFP info
                "source": source or "RFP_Processing",
                "source_type": "completed_rfp",
                "rfp_name": rfp_path.stem,  # filename without extension
                "rfp_file": rfp_path.name,
                "rfp_path": str(rfp_path),
                "rfp_age": rfp_age,  # RFP counter/age
                "submitter_name": submitter_name,
                
                # Question data
                "question": question,
                "answer": answer,
                "comments": comments,
                "validator_name": validator_name,
                "question_index": index,
                
                # Technical metadata  
                "indexed_at": datetime.now().isoformat(),
                "document_type": "rfp_qa_pair",
                "embedded_field": "question"
            }
            
            payloads.append(metadata)
        
        if not questions:
            print("❌ No valid questions found for indexing")
            return 0
        
        # Validate vector count matches question count
        if len(valid_vectors) != len(questions):
            raise ValueError(f"Vector count mismatch: {len(valid_vectors)} vectors != {len(questions)} questions")
        
        print(f"✅ Extracted {len(questions)} valid Q&A pairs with matching vectors")
        print(f"🗃️ Indexing to {collection_name} with RFP age #{rfp_age}...")
        
        # Build points using provided vectors
        points = [
            PointStruct(id=str(uuid4()), vector=vec, payload=pld)
            for vec, pld in zip(valid_vectors, payloads)
        ]
        
        # Upsert to Qdrant
        client = get_qdrant_client()
        client.upsert(collection_name=collection_name, points=points)
        indexed_count = len(points)
        
        print(f"✅ Successfully indexed {indexed_count} Q&A pairs")
        print(f"📊 Metadata stored: submitter={submitter_name}, source={source or 'Not specified'}, validator_count={len([p for p in payloads if p['validator_name']])}")
        
        # Perform automatic cleanup after successful indexing
        if auto_cleanup and indexed_count > 0:
            print(f"🧹 Running automatic cleanup for RFP age #{rfp_age}...")
            cleanup_count = tracker.cleanup_old_rfps()
            if cleanup_count > 0:
                print(f"✅ Cleaned up {cleanup_count} old RFP documents")
        
        return indexed_count
        
    except Exception as e:
        print(f"❌ Error parsing RFP: {e}")
        raise
