# 🚀 AI Agent RFP Management System

Un système intelligent de gestion des RFP (Request for Proposal) utilisant l'IA et les bases de données vectorielles pour automatiser et améliorer le processus de réponse aux appels d'offres.

## 🎯 Fonctionnalités Principales

### 📋 Gestion des RFP
- **Parsing automatique** : Extraction intelligente des questions depuis Excel/PDF
- **Interface Streamlit** : Interface utilisateur intuitive pour la gestion des réponses
- **Validation humaine** : Système de validation avec traçabilité des validateurs
- **Export automatisé** : Génération de fichiers Excel avec réponses complètes

### 🧠 IA et Recherche Vectorielle
- **Triple Retrieval** : Recherche simultanée dans 3 sources :
  - 📄 Documentation interne (knowledge base)
  - 🔍 Historique des Q&A validées
  - 🌐 Recherche web contextuelle (DuckDuckGo)
- **ReAct Integration** : Agent IA avec raisonnement et actions
- **Embeddings sémantiques** : Utilisation de `sentence-transformers/all-MiniLM-L6-v2`

### 🗄️ Base de Données Vectorielle (Qdrant)
- **Collections multiples** : Séparation logique des données
- **Recherche sémantique** : Similarité cosinus sur vecteurs 384D
- **Métadonnées enrichies** : Traçabilité complète (timestamps, validateurs)

## 🏗️ Architecture

```
AI_AGENT_PROJECT/
├── 📁 parsers/          # Parsing RFP et documentation
│   ├── rfp_parser.py    # Parser principal RFP
│   └── internal_parser.py # Parser documentation interne
├── 📁 qdrant/           # Gestion base vectorielle
│   ├── client.py        # Client Qdrant
│   ├── indexer.py       # Indexation documents
│   ├── retriever.py     # Triple retrieval
│   ├── react_retriever.py # Agent ReAct
│   ├── inspector.py     # Inspection collections
│   └── cleaner.py       # Maintenance DB
├── 📁 ui/               # Interface utilisateur
│   └── rfp_manager.py   # Interface Streamlit
├── 📁 outputs/          # Fichiers générés
└── settings.py          # Configuration
```

## 🔧 Installation

### Prérequis
```bash
Python 3.8+
pip install -r requirements.txt
```

### Configuration
1. **Copier le fichier d'environnement** :
```bash
cp .env.example .env
```

2. **Configurer Qdrant** :
```env
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-api-key
```

3. **Activer l'environnement virtuel** :
```bash
source venv/bin/activate  # Linux/Mac
```

## 🚀 Utilisation

### Interface Principale
```bash
streamlit run ui/rfp_manager.py
```

### Outils VectorDB

#### 🔍 Inspection des Collections
```bash
python qdrant/inspector.py
```

#### 🧹 Nettoyage de la Base
```bash
python qdrant/cleaner.py
# ou mode batch
python qdrant/cleaner.py --quick-clean
```

#### 🤖 Test ReAct Agent
```bash
python react_demo_simple.py
```

## 📊 Collections Qdrant

| Collection | Points | Dimension | Usage |
|------------|--------|-----------|-------|
| `internal_knowledge_base` | 1578+ | 384D | Documentation technique interne |
| `rfp_qa_history` | 477+ | 384D | Historique Q&A validées |

## 🔄 Workflow Complet

### 1. Upload et Parsing → 2. Recherche Triple → 3. Formulation → 4. Validation → 5. Export

## 🤝 Contribution

### Structure de Commit
```
🚀 feat: nouvelle fonctionnalité
🐛 fix: correction de bug  
📚 docs: documentation
```

## 📄 Licence

MIT License - Développé avec ❤️ pour automatiser les réponses aux RFP