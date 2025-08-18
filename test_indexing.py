#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de test pour les nouvelles fonctions d'indexation
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


def test_internal_data_indexing():
    """Test l'indexation des documents internes"""
    print("🔍 Test d'indexation des documents internes...")
    
    # Chemin vers les documents internes (vous pouvez ajuster)
    internal_data_path = project_root / "data" / "processed"
    
    if internal_data_path.exists():
        try:
            count = index_internal_data_with_source(
                str(internal_data_path), 
                source_name="okta_documentation"
            )
            print(f"✅ {count} documents internes indexés avec succès")
        except Exception as e:
            print(f"❌ Erreur lors de l'indexation des documents internes: {e}")
    else:
        print(f"📁 Dossier {internal_data_path} non trouvé")


def test_rfp_indexing():
    """Test l'indexation d'un RFP complété"""
    print("\n🔍 Test d'indexation d'un RFP complété...")
    
    # Chercher des fichiers Excel dans le dossier outputs
    outputs_path = project_root / "outputs"
    
    if outputs_path.exists():
        excel_files = list(outputs_path.glob("*.xlsx"))
        if excel_files:
            excel_file = excel_files[0]  # Prendre le premier fichier trouvé
            print(f"📄 Test avec le fichier: {excel_file.name}")
            
            source_info = {
                "client_name": "Test Client",
                "project": "Test RFP Project",
                "completion_date": "2025-08-18",
                "completed_by": "AI Agent",
                "test_run": True
            }
            
            try:
                count = index_completed_rfp(str(excel_file), source_info)
                print(f"✅ {count} paires Q&A indexées avec succès")
            except Exception as e:
                print(f"❌ Erreur lors de l'indexation du RFP: {e}")
        else:
            print("📁 Aucun fichier Excel trouvé dans outputs/")
    else:
        print(f"📁 Dossier outputs non trouvé")


def test_batch_rfp_indexing():
    """Test l'indexation en lot des RFPs complétés"""
    print("\n🔍 Test d'indexation en lot des RFPs complétés...")
    
    completed_rfps_path = project_root / "data" / "completed_RFPs"
    
    if completed_rfps_path.exists():
        try:
            total = batch_index_completed_rfps(str(completed_rfps_path))
            print(f"✅ {total} paires Q&A indexées au total")
        except Exception as e:
            print(f"❌ Erreur lors de l'indexation en lot: {e}")
    else:
        print(f"📁 Dossier {completed_rfps_path} non trouvé")


def main():
    print("🚀 Test des nouvelles fonctions d'indexation")
    print("=" * 50)
    
    # Test 1: Documents internes
    test_internal_data_indexing()
    
    # Test 2: RFP individuel
    test_rfp_indexing()
    
    # Test 3: RFPs en lot
    test_batch_rfp_indexing()
    
    print("\n✅ Tests terminés !")


if __name__ == "__main__":
    main()
