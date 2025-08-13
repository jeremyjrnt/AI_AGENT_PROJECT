#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test simple du pipeline RAG : Parser → Qdrant → Indexer → Vérification
"""

from parsers.internal_parser import parse_folder_to_data, save_parsed_data
from qdrant.client import setup_collections_dynamic, get_qdrant_client, INTERNAL_COLLECTION
from qdrant.indexer import index_internal_documents
import settings


def main():
    print("🚀 TEST PIPELINE RAG")
    print("=" * 40)
    print(f"🔧 Provider: {settings.EMBEDDING_PROVIDER}")
    print(f"🔧 Modèle: {settings.EMBEDDING_MODEL}")
    print()
    
    # 1. PARSING HTML avec votre fonction
    folder_path = r"C:\Users\edens\OneDrive - Technion\Aviv-25\DataScienceApp\AI_AGENT_PROJECT\data\help.okta.com"
    print(f"📂 Chemin: {folder_path}")
    data = parse_folder_to_data(folder_path)
    
    if not data:
        print("❌ Aucune donnée parsée")
        return
    
    print(f"✅ {len(data)} chunks parsés")
    save_parsed_data(data, "parsed_data.json")
    
    # Afficher exemple
    print(f"\n📄 Exemple de chunk:")
    chunk = data[0]
    print(f"  - text: {chunk.get('text', '')[:100]}...")
    print(f"  - metadata: {chunk.get('metadata', {})}")
    
    # 2. SETUP QDRANT avec votre fonction  
    print(f"\n🔧 Configuration Qdrant...")
    vector_size = setup_collections_dynamic(reset_rfp_history=False)
    print(f"✅ Collections OK (vector_size: {vector_size})")
    
    # 3. INDEXATION avec votre fonction
    print(f"\n📤 Indexation dans Qdrant...")
    
    # Format pour votre indexer
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
    print(f"✅ {count} documents indexés")
    
    # 4. VÉRIFICATION avec votre client
    print(f"\n🔍 Vérification...")
    client = get_qdrant_client()
    collection_info = client.get_collection(INTERNAL_COLLECTION)
    points_count = collection_info.points_count
    
    print(f"✅ Collection '{INTERNAL_COLLECTION}': {points_count} points")
    
    # Voir quelques points
    if points_count > 0:
        result = client.scroll(
            collection_name=INTERNAL_COLLECTION,
            limit=2,
            with_payload=True,
            with_vectors=False
        )
        
        points = result[0]
        print(f"\n📄 Exemples de points stockés:")
        
        for i, point in enumerate(points, 1):
            print(f"\n  Point {i}:")
            print(f"    - Text: {point.payload.get('text', '')[:80]}...")
            print(f"    - Title: {point.payload.get('title', 'N/A')}")
            print(f"    - File: {point.payload.get('file_name', 'N/A')}")
    
    print(f"\n🎉 PIPELINE TERMINÉ !")
    print(f"✅ {len(data)} chunks → {count} indexés → {points_count} dans Qdrant")


if __name__ == "__main__":
    main()
