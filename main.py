#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test complet du pipeline RAG : Parser → Qdrant → Embedding → Indexer → Vérification
"""

from parsers.internal_parser import parse_folder_to_data, save_parsed_data
from qdrant.client import setup_collections_dynamic, get_qdrant_client, get_embeddings
from qdrant.indexer import index_internal_documents
import settings


def test_parsing(folder_path: str):
    """Test du parsing HTML"""
    print("� ÉTAPE 1: PARSING HTML")
    print("=" * 50)
    
    data = parse_folder_to_data(folder_path)
    
    if not data:
        print("❌ Aucune donnée parsée")
        return None
    
    print(f"✅ {len(data)} chunks parsés")
    
    # Afficher un exemple de chunk
    if data:
        print("\n📄 Exemple de chunk parsé:")
        chunk = data[0]
        print(f"  - chunk_id: {chunk.get('chunk_id', 'N/A')}")
        print(f"  - title: {chunk.get('title', 'N/A')}")
        print(f"  - text: {chunk.get('text', '')[:100]}...")
        print(f"  - metadata: {chunk.get('metadata', {})}")
    
    # Sauvegarder pour référence
    save_parsed_data(data, "parsed_data.json")
    print(f"💾 Données sauvées dans parsed_data.json")
    
    return data


def test_qdrant_setup():
    """Test de la configuration Qdrant"""
    print("\n🔧 ÉTAPE 2: CONFIGURATION QDRANT")
    print("=" * 50)
    
    try:
        # Configuration des collections
        vector_size = setup_collections_dynamic(reset_rfp_history=False)
        print(f"✅ Collections configurées (vector_size: {vector_size})")
        
        # Test de connexion
        client = get_qdrant_client()
        collections = client.get_collections()
        print(f"✅ Connexion Qdrant OK - {len(collections.collections)} collection(s)")
        
        for col in collections.collections:
            print(f"  - {col.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur Qdrant: {e}")
        return False


def test_embedding():
    """Test de génération d'embeddings"""
    print("\n🧠 ÉTAPE 3: TEST EMBEDDINGS")
    print("=" * 50)
    
    try:
        embeddings_provider = get_embeddings()
        print(f"✅ Provider: {type(embeddings_provider).__name__}")
        
        # Test avec un texte simple
        test_text = "Ceci est un test d'embedding"
        vectors = embeddings_provider.embed_documents([test_text])
        
        print(f"✅ Embedding généré - Dimension: {len(vectors[0])}")
        print(f"✅ Premier values: {vectors[0][:5]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur embedding: {e}")
        return False


def test_indexing(data):
    """Test d'indexation dans Qdrant"""
    print("\n📤 ÉTAPE 4: INDEXATION DANS QDRANT")
    print("=" * 50)
    
    try:
        # Préparer les documents pour l'indexer
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
        
        # Indexer (prendre seulement les 5 premiers pour le test)
        test_docs = docs_for_indexer[:5]
        count = index_internal_documents(test_docs)
        
        print(f"✅ {count} documents indexés avec succès")
        return True
        
    except Exception as e:
        print(f"❌ Erreur indexation: {e}")
        return False


def test_retrieval_verification():
    """Vérification que les données sont bien dans Qdrant"""
    print("\n🔍 ÉTAPE 5: VÉRIFICATION DANS QDRANT")
    print("=" * 50)
    
    try:
        from qdrant.client import INTERNAL_COLLECTION
        
        client = get_qdrant_client()
        
        # Info sur la collection
        collection_info = client.get_collection(INTERNAL_COLLECTION)
        points_count = collection_info.points_count
        vector_size = collection_info.config.params.vectors.size
        
        print(f"✅ Collection '{INTERNAL_COLLECTION}':")
        print(f"  - Nombre de points: {points_count}")
        print(f"  - Taille des vecteurs: {vector_size}")
        
        # Récupérer quelques points pour vérifier le contenu
        if points_count > 0:
            result = client.scroll(
                collection_name=INTERNAL_COLLECTION,
                limit=2,
                with_payload=True,
                with_vectors=False
            )
            
            points = result[0]
            print(f"\n📄 Exemple de points stockés:")
            
            for i, point in enumerate(points, 1):
                print(f"\n  Point {i}:")
                print(f"    - ID: {point.id}")
                print(f"    - Text: {point.payload.get('text', '')[:100]}...")
                print(f"    - Title: {point.payload.get('title', 'N/A')}")
                print(f"    - File: {point.payload.get('file_name', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur vérification: {e}")
        return False


def main():
    print("🚀 TEST COMPLET DU PIPELINE RAG")
    print("=" * 60)
    print(f"🔧 Provider d'embeddings: {settings.EMBEDDING_PROVIDER}")
    print(f"🔧 Modèle: {settings.EMBEDDING_MODEL}")
    print()
    
    # Chemin du dossier à parser
    folder_path = input("📂 Chemin du dossier HTML: ")
    
    # Pipeline complet
    success_count = 0
    
    # 1. Parsing
    data = test_parsing(folder_path)
    if data:
        success_count += 1
    else:
        print("❌ Arrêt: parsing échoué")
        return
    
    # 2. Configuration Qdrant
    if test_qdrant_setup():
        success_count += 1
    else:
        print("❌ Arrêt: configuration Qdrant échouée")
        return
    
    # 3. Test embedding
    if test_embedding():
        success_count += 1
    else:
        print("❌ Arrêt: embedding échoué")
        return
    
    # 4. Indexation
    if test_indexing(data):
        success_count += 1
    else:
        print("❌ Arrêt: indexation échouée")
        return
    
    # 5. Vérification
    if test_retrieval_verification():
        success_count += 1
    
    # Résumé final
    print("\n" + "=" * 60)
    print("🎯 RÉSUMÉ FINAL")
    print("=" * 60)
    print(f"✅ Étapes réussies: {success_count}/5")
    
    if success_count == 5:
        print("🎉 PIPELINE COMPLET RÉUSSI !")
        print("✅ Vos données HTML sont parsées, embeddées et stockées dans Qdrant")
    else:
        print("⚠️ Certaines étapes ont échoué")


if __name__ == "__main__":
    main()
