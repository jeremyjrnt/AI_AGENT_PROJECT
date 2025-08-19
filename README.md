# AI Agent RFP Management System

Un système intelligent de gestion des RFP (Request for Proposal) utilisant l'IA et les bases de données vectorielles pour automatiser et améliorer le processus de réponse aux appels d'offres.

## 🎯 Fonctionnalités Principales

### 📋 Gestion des RFP
- **Interface Professionnelle** : Interface Streamlit élégante sans emojis, avec design moderne et onglets
- **Parsing automatique** : Extraction intelligente des questions depuis Excel/PDF
- **Numérotation séquentielle** : Système de numérotation automatique des RFP avec nettoyage automatique
- **Validation humaine** : Système de validation avec traçabilité des validateurs
- **Export automatisé** : Génération de fichiers Excel avec réponses complètes

### 🧠 IA et Recherche Vectorielle
- **Triple Retrieval** : Recherche simultanée dans 3 sources :
  - 📄 Documentation interne (knowledge base)
  - 🔍 Historique des Q&A validées
  - 🌐 Recherche web contextuelle (DuckDuckGo)
- **Modes AI Flexibles** : 
  - Mode Développement (Ollama gratuit)
  - Mode Production (OpenAI haute qualité)
- **ReAct Integration** : Agent IA avec raisonnement et actions
- **Embeddings sémantiques** : Utilisation de `sentence-transformers/all-MiniLM-L6-v2`

### 🗄️ Base de Données Vectorielle (Qdrant)
- **Collections multiples** : Séparation logique des données
- **Recherche sémantique** : Similarité cosinus sur vecteurs 384D
- **Métadonnées enrichies** : Traçabilité complète (timestamps, validateurs, numéros RFP)
- **Auto-nettoyage** : Suppression automatique des anciens RFP (configurable)

### 🎨 Interface Utilisateur
- **Design Professionnel** : Interface moderne sans emojis, adaptée aux environnements corporate
- **Onglets Séparés** : "RFP Manager" et "Guide Utilisateur" pour une navigation claire
- **Styling CSS Personnalisé** : Effets de survol, animations douces, palette de couleurs professionnelle
- **Responsive** : Interface adaptative avec sidebar et layout large

### 🔧 Outils de Gestion
- **CLI Management** : `rfp_manager_cli.py` pour gestion en ligne de commande
- **Statistiques RFP** : Suivi des métriques et de l'état de la base vectorielle
- **Inspection Collections** : Outils de debug et visualisation des données
- **Reset/Cleanup** : Fonctions de maintenance et réinitialisation

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
│   ├── rfp_tracker.py   # Numérotation et tracking RFP
│   ├── inspector.py     # Inspection collections
│   └── cleaner.py       # Maintenance DB
├── 📁 ui/               # Interface utilisateur
│   └── rfp_manager.py   # Interface Streamlit professionnelle
├── 📁 outputs/          # Fichiers générés
├── rfp_manager_cli.py   # Outils CLI de gestion
├── remove_emojis.py     # Script de nettoyage interface
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

## ✨ Nouvelles Fonctionnalités (Août 2025)

### Interface Professionnelle
- **Design Corporate** : Interface entièrement redesignée sans emojis pour un usage professionnel
- **Styling Moderne** : CSS personnalisé avec onglets élégants, effets de survol et palette de couleurs cohérente
- **Expérience Utilisateur** : Navigation améliورée avec séparation claire entre workflow et documentation

### Système RFP Avancé
- **Numérotation Séquentielle** : Attribution automatique de numéros uniques aux RFP
- **Auto-cleanup** : Suppression automatique des anciens RFP basée sur l'âge (configurable)
- **Tracking Complet** : Suivi des statistiques et métriques RFP en temps réel

### Outils de Gestion
- **CLI Management** : Interface en ligne de commande pour administration système
- **Inspection Avancée** : Outils de debug et visualisation des collections vectorielles
- **Reset/Maintenance** : Fonctions de réinitialisation et nettoyage des données

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