#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple RAG pipeline test: Parser → Qdrant → Indexer → Verification
Moved from project root to test/ folder.
"""

from parsers.internal_parser import parse_folder_to_data, save_parsed_data
from qdrant.client import setup_collections_dynamic, get_qdrant_client, INTERNAL_COLLECTION
from qdrant.indexer import index_internal_documents
import settings


def test_rag_pipeline():
    print("🚀 TEST PIPELINE RAG")
    print("=" * 40)
    print(f"🔧 Provider: {settings.EMBEDDING_PROVIDER}")
    print(f"🔧 Model: {settings.EMBEDDING_MODEL}")
    print()
    
    # 1. PARSING HTML
    import sys
    folder_path = sys.argv[1] if len(sys.argv) > 1 else r"data/okta_doc"
    print(f"📂 Path: {folder_path}")
    data = parse_folder_to_data(folder_path)
    
    if not data:
        print("❌ No data parsed")
        return
    
    print(f"✅ {len(data)} chunks parsed")
    save_parsed_data(data, "parsed_data.json")
    
    # Show example
    print(f"\n📄 Example chunk:")
    chunk = data[0]
    print(f"  - text: {chunk.get('text', '')[:100]}...")
    print(f"  - metadata: {chunk.get('metadata', {})}")
    
    # 2. SETUP QDRANT
    print(f"\n🔧 Qdrant setup...")
    vector_size = setup_collections_dynamic(reset_rfp_history=False)
    print(f"✅ Collections OK (vector_size: {vector_size})")
    
    # 3. INDEXING
    print(f"\n📤 Indexing in Qdrant...")
    docs_for_indexer = []
    for chunk in data:
        doc = {
            "text": chunk["text"],
            "metadata": {
                "chunk_id": chunk.get("chunk_id"),
                "doc_id": chunk.get("doc_id"), 
                "title": chunk.get("title"),
                "source": chunk.get("source", "internal"),
                "file_name": chunk.get("metadata", {}).get("file_name"),
                "file_path": chunk.get("metadata", {}).get("file_path")
            }
        }
        docs_for_indexer.append(doc)
    
    count = index_internal_documents(docs_for_indexer)
    print(f"✅ {count} documents indexed")
    
    # 4. VERIFICATION
    print(f"\n🔍 Verification...")
    client = get_qdrant_client()
    collection_info = client.get_collection(INTERNAL_COLLECTION)
    points_count = collection_info.points_count
    
    print(f"✅ Collection '{INTERNAL_COLLECTION}': {points_count} points")
    
    # Show some points
    if points_count > 0:
        result = client.scroll(
            collection_name=INTERNAL_COLLECTION,
            limit=5,
            with_payload=True,
            with_vectors=False
        )

        points = result[0]
        print(f"\n📄 Example stored points:")

        for i, point in enumerate(points, 1):
            print(f"\n  Point {i}:")
            print(f"    - Text: {point.payload.get('text', '')[:80]}...")
            print(f"    - Title: {point.payload.get('title', 'N/A')}")
            print(f"    - File: {point.payload.get('file_name', 'N/A')}")
    
    print(f"\n🎉 PIPELINE FINISHED!")
    print(f"✅ {len(data)} chunks → {count} indexed → {points_count} in Qdrant")


if __name__ == "__main__":
    test_rag_pipeline()
