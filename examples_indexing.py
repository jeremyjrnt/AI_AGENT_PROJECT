#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exemples d'utilisation des nouvelles fonctions d'indexation
"""

import sys
from pathlib import Path

# Ajouter le répertoire racine au path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from qdrant.indexer import (
    index_internal_data_with_source, 
    index_completed_rfp,
    batch_index_completed_rfps
)


def example_1_index_internal_docs():
    """
    Exemple 1: Indexer des documents internes
    """
    print("📚 Exemple 1: Indexation de documents internes")
    print("-" * 50)
    
    # Chemin vers un dossier contenant des documents internes
    internal_docs_path = "/path/to/your/internal/docs"
    
    # Exemple avec des métadonnées personnalisées
    try:
        count = index_internal_data_with_source(
            data_folder_path=internal_docs_path,
            source_name="company_policies_2025"
        )
        print(f"✅ {count} documents internes indexés")
        
        # Les documents seront indexés avec ces métadonnées :
        # - source: "internal"
        # - source_type: "internal_docs"  
        # - source_name: "company_policies_2025"
        # - file_path: chemin complet du fichier
        # - file_name: nom du fichier
        # - indexed_at: timestamp
        # - document_type: "internal_documentation"
        
    except FileNotFoundError:
        print("📁 Dossier d'exemple non trouvé (normal pour cet exemple)")


def example_2_index_single_rfp():
    """
    Exemple 2: Indexer un RFP complété
    """
    print("\n📋 Exemple 2: Indexation d'un RFP complété")
    print("-" * 50)
    
    # Chemin vers un fichier Excel RFP complété
    rfp_excel_path = "/path/to/completed/rfp.xlsx"
    
    # Métadonnées personnalisées sur la source
    source_info = {
        "client_name": "TechCorp Inc",
        "project": "Cloud Migration RFP 2025",
        "completion_date": "2025-08-18",
        "completed_by": "John Doe",
        "rfp_value": "$2.5M",
        "industry": "Technology",
        "region": "North America"
    }
    
    try:
        count = index_completed_rfp(rfp_excel_path, source_info)
        print(f"✅ {count} paires Q&A indexées")
        
        # Chaque paire Q&A sera indexée avec :
        # - Vecteur créé à partir de la QUESTION
        # - Métadonnées contenant :
        #   - source: "past-rfp"
        #   - source_type: "completed_rfp"
        #   - question: texte de la question
        #   - answer: réponse (Yes/No)
        #   - comment: commentaire associé
        #   - + toutes les métadonnées de source_info
        
    except FileNotFoundError:
        print("📁 Fichier RFP d'exemple non trouvé (normal pour cet exemple)")


def example_3_batch_index_rfps():
    """
    Exemple 3: Indexer tous les RFPs d'un dossier
    """
    print("\n📦 Exemple 3: Indexation en lot de RFPs")
    print("-" * 50)
    
    # Dossier contenant plusieurs fichiers Excel de RFPs complétés
    completed_rfps_folder = project_root / "data" / "completed_RFPs"
    
    try:
        total_indexed = batch_index_completed_rfps(str(completed_rfps_folder))
        print(f"✅ {total_indexed} paires Q&A indexées au total")
        
        # Cette fonction :
        # - Traite automatiquement tous les fichiers .xlsx du dossier
        # - Extrait des infos depuis les noms de fichiers si possible
        # - Ajoute des métadonnées de lot (batch_indexed, batch_date, etc.)
        
    except FileNotFoundError:
        print("📁 Dossier completed_RFPs non trouvé (normal si vide)")


def example_4_search_indexed_data():
    """
    Exemple 4: Comment chercher dans les données indexées
    """
    print("\n🔍 Exemple 4: Recherche dans les données indexées")
    print("-" * 50)
    
    print("""
    Une fois les données indexées, vous pouvez les rechercher avec :
    
    from qdrant.retriever import search_documents
    from qdrant.client import INTERNAL_COLLECTION, RFP_QA_COLLECTION
    
    # Rechercher dans les documents internes
    internal_results = search_documents(
        query="sécurité des données",
        collection_name=INTERNAL_COLLECTION,
        top_k=5
    )
    
    # Rechercher dans les RFPs passés
    rfp_results = search_documents(
        query="authentification multi-facteurs",
        collection_name=RFP_QA_COLLECTION,
        top_k=5
    )
    
    # Filtrer par source
    filtered_results = search_documents(
        query="backup procedures",
        collection_name=RFP_QA_COLLECTION,
        top_k=10,
        filter_conditions={"source": "past-rfp", "client_name": "TechCorp Inc"}
    )
    """)


def main():
    print("🚀 Exemples d'utilisation des fonctions d'indexation")
    print("=" * 60)
    
    example_1_index_internal_docs()
    example_2_index_single_rfp()  
    example_3_batch_index_rfps()
    example_4_search_indexed_data()
    
    print("\n" + "=" * 60)
    print("📖 Guide d'utilisation :")
    print("1. Préparez vos documents internes (txt, md) dans un dossier")
    print("2. Complétez vos RFPs et sauvez-les en Excel avec colonnes Question/Answer/Comment")
    print("3. Utilisez les fonctions pour indexer dans Qdrant")
    print("4. Recherchez avec les outils de retrieval pour la pré-completion")
    print("\n✨ Les métadonnées permettent de filtrer et tracer les sources !")


if __name__ == "__main__":
    main()
