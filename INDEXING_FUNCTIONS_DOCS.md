# 📚 Nouvelles Fonctions d'Indexation - Documentation

## 🎯 Fonctions Ajoutées dans `qdrant/indexer.py`

### 1. `index_internal_data_with_source(data_folder_path, source_name)`

**Objectif :** Indexer des documents internes dans la base vectorielle avec métadonnées de source.

**Paramètres :**
- `data_folder_path` (str) : Chemin vers le dossier contenant les documents
- `source_name` (str) : Nom pour identifier la source (ex: "company_handbook")

**Formats supportés :** `.txt`, `.md`

**Métadonnées automatiques :**
```json
{
  "source": "internal",
  "source_type": "internal_docs", 
  "source_name": "nom_fourni",
  "file_path": "/chemin/complet/fichier.txt",
  "file_name": "fichier.txt",
  "indexed_at": "2025-08-18T10:30:00",
  "document_type": "internal_documentation"
}
```

**Exemple d'utilisation :**
```python
from qdrant.indexer import index_internal_data_with_source

count = index_internal_data_with_source(
    "/path/to/internal/docs", 
    "okta_documentation"
)
print(f"{count} documents indexés")
```

---

### 2. `index_completed_rfp(rfp_excel_path, rfp_source_info)`

**Objectif :** Indexer un RFP complété où les questions deviennent des vecteurs et les réponses des métadonnées.

**Paramètres :**
- `rfp_excel_path` (str) : Chemin vers le fichier Excel RFP complété
- `rfp_source_info` (dict, optionnel) : Informations supplémentaires sur la source

**Format Excel attendu :**
- Colonnes requises : Question, Answer
- Colonne optionnelle : Comment/Note

**Le vecteur est créé depuis :** La QUESTION  
**Les métadonnées incluent :** La réponse + infos de source

**Métadonnées automatiques :**
```json
{
  "source": "past-rfp",
  "source_type": "completed_rfp",
  "question": "Texte de la question",
  "answer": "Yes/No",
  "comment": "Commentaire associé",
  "rfp_file": "nom_fichier.xlsx",
  "rfp_path": "/chemin/complet/fichier.xlsx", 
  "question_index": 0,
  "indexed_at": "2025-08-18T10:30:00",
  "document_type": "rfp_qa_pair"
}
```

**Exemple d'utilisation :**
```python
from qdrant.indexer import index_completed_rfp

source_info = {
    "client_name": "TechCorp",
    "project": "Cloud Migration RFP",
    "completion_date": "2025-08-18",
    "rfp_value": "$2.5M"
}

count = index_completed_rfp(
    "/path/to/completed_rfp.xlsx", 
    source_info
)
print(f"{count} paires Q&A indexées")
```

---

### 3. `batch_index_completed_rfps(completed_rfps_folder)`

**Objectif :** Indexer en lot tous les RFPs Excel d'un dossier.

**Paramètres :**
- `completed_rfps_folder` (str) : Chemin vers le dossier contenant les fichiers Excel

**Fonctionnalités :**
- Traite automatiquement tous les `.xlsx` du dossier
- Extrait des informations depuis les noms de fichiers si possible
- Ajoute des métadonnées de traitement en lot

**Métadonnées supplémentaires :**
```json
{
  "batch_indexed": true,
  "batch_date": "2025-08-18T10:30:00", 
  "source_folder": "/path/to/folder",
  "derived_client": "extrait_du_nom_fichier",
  "derived_project": "extrait_du_nom_fichier"
}
```

**Exemple d'utilisation :**
```python
from qdrant.indexer import batch_index_completed_rfps

total = batch_index_completed_rfps("/data/completed_RFPs/")
print(f"{total} paires Q&A indexées au total")
```

---

## 🔍 Utilisation avec la Recherche

Une fois indexées, les données peuvent être recherchées :

### Recherche dans documents internes :
```python
from qdrant.client import INTERNAL_COLLECTION
from qdrant.retriever import search_documents

results = search_documents(
    query="politique de sécurité",
    collection_name=INTERNAL_COLLECTION,
    top_k=5
)
```

### Recherche dans RFPs passés :
```python  
from qdrant.client import RFP_QA_COLLECTION

results = search_documents(
    query="authentification multi-facteurs", 
    collection_name=RFP_QA_COLLECTION,
    top_k=5,
    filter_conditions={"source": "past-rfp", "client_name": "TechCorp"}
)
```

---

## 📁 Structure des Collections

### INTERNAL_COLLECTION
- **Contenu :** Documents internes de l'entreprise
- **Vecteurs :** Texte complet des documents  
- **Source :** `"internal"`

### RFP_QA_COLLECTION  
- **Contenu :** Paires Question/Réponse des RFPs passés
- **Vecteurs :** Questions RFP
- **Métadonnées :** Réponses + contexte client/projet
- **Source :** `"past-rfp"`

---

## ✅ Avantages

1. **Traçabilité complète** : Métadonnées détaillées pour chaque source
2. **Recherche ciblée** : Filtrage par source, client, projet
3. **Historique RFP** : Réutilisation des réponses passées
4. **Scalabilité** : Traitement en lot pour grandes quantités
5. **Flexibilité** : Métadonnées personnalisables selon les besoins

---

## 🧪 Tests Disponibles

- `test_indexing.py` : Tests automatiques des fonctions
- `examples_indexing.py` : Exemples d'utilisation détaillés

**Commande de test :**
```bash
python test_indexing.py
```
