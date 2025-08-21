# RFPilot - AI Agent RFP Management System

Enterprise-grade Request for Proposal (RFP) automation system leveraging advanced AI, vector databases, and ReAct agents to streamline proposal response workflows.

## Overview

This system provides intelligent automation for RFP processing through semantic search, automated question answering, and comprehensive document management. Built with modern AI technologies including Azure OpenAI, Qdrant vector database, and LangChain ReAct agents.

## Core Features

### RFP Processing Pipeline

- **Intelligent Document Parsing**: Automated extraction of questions from Excel and PDF documents
- **Sequential Numbering System**: Automatic RFP numbering with age-based cleanup mechanisms
- **Human Validation Workflow**: Structured validation process with complete audit trails
- **Automated Export**: Generation of comprehensive Excel reports with completed responses

### AI-Powered Question Answering

- **ReAct Agent Architecture**: Reasoning and Acting agents with structured tool usage
- **Triple Retrieval System**: Simultaneous search across three knowledge sources:
  - Internal knowledge base documentation
  - Historical Q&A validated responses
  - Contextual web search integration
- **Vector Pre-calculation**: Performance-optimized embedding storage and reuse
- **Multi-Model Support**: Azure OpenAI (production) and Ollama (development) configurations

### Vector Database Management

- **Qdrant Integration**: High-performance vector similarity search
- **Multiple Collections**: Logical separation of knowledge domains
- **Semantic Search**: Cosine similarity on 384D or 3072D vectors
- **Metadata Enrichment**: Complete traceability with timestamps, validators, and RFP identifiers
- **Automatic Cleanup**: Configurable age-based document removal system

### Professional User Interface

- **Corporate Design**: Clean, professional interface suitable for enterprise environments
- **Modular Architecture**: Separated workflow and documentation interfaces
- **Advanced Styling**: Custom CSS with hover effects and professional color schemes
- **Responsive Layout**: Adaptive design with optimized sidebar and wide layouts

### Management Tools

- **Command-Line Interface**: Comprehensive CLI for system administration
- **Analytics Dashboard**: Real-time metrics and vector database statistics
- **Collection Inspector**: Advanced debugging and data visualization tools
- **Maintenance Suite**: Reset, cleanup, and optimization utilities

## Technical Architecture

```
AI_AGENT_PROJECT/
├── parsers/             # Document processing modules
│   ├── rfp_parser.py    # Primary RFP document parser
│   └── internal_parser.py # Internal documentation processor
├── qdrant/              # Vector database management
│   ├── client.py        # Qdrant client configuration
│   ├── indexer.py       # Document indexing system
│   ├── retriever.py     # Multi-source retrieval engine
│   ├── react_retriever.py # ReAct agent implementation
│   ├── rfp_tracker.py   # RFP lifecycle management
│   ├── inspector.py     # Collection inspection tools
│   └── cleaner.py       # Database maintenance utilities
├── ui/                  # User interface components
│   └── rfp_manager.py   # Streamlit application interface
├── outputs/             # Generated documents and reports
├── data/                # Document storage and processing
├── test/                # Testing suite and utilities
├── settings.py          # Configuration management
└── requirements.txt     # Python dependencies
```

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- Virtual environment management
- Access to Qdrant vector database
- Azure OpenAI or OpenAI API credentials (optional)

### Environment Setup

1. **Clone and prepare environment**:

```bash
git clone <repository-url>
cd AI_AGENT_PROJECT
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Configure environment variables**:

```bash
cp .env.example .env
```

### Configuration Settings

#### Essential Configuration (.env file)

```env
# Qdrant Vector Database
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-api-key

# Azure OpenAI (Production)
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_CHAT_DEPLOYMENT=team11-gpt4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=team11-embedding

# Standard OpenAI (Alternative)
OPENAI_API_KEY=your-openai-key
OPENAI_EMBEDDING_MODEL=text-embedding-3-large

# Embedding Provider Selection
EMBEDDING_PROVIDER=huggingface  # or "openai"
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## Usage Guide

### Primary Interface

Launch the main application interface:

```bash
streamlit run ui/rfp_manager.py
```

### Command-Line Tools

#### Vector Database Inspection

```bash
python qdrant/inspector.py
```

#### Database Maintenance

```bash
python qdrant/cleaner.py
python qdrant/cleaner.py --quick-clean
```

#### ReAct Agent Testing

```bash
python react_demo_simple.py
```

#### CLI Management Interface

```bash
python rfp_manager_cli.py
```

## System Specifications

### Vector Database Collections

| Collection Name             | Document Count | Vector Dimension | Primary Usage           |
| --------------------------- | -------------- | ---------------- | ----------------------- |
| `internal_knowledge_base` | 1,500+         | 384D/3072D       | Technical documentation |
| `rfp_qa_history`          | 450+           | 384D/3072D       | Validated Q&A responses |

### Performance Characteristics

- **Query Response Time**: < 2 seconds average
- **Vector Search**: Cosine similarity with 95%+ accuracy
- **Concurrent Users**: Supports 10+ simultaneous sessions
- **Document Processing**: 100+ questions per batch
- **Memory Footprint**: < 2GB RAM typical usage

### Integration Capabilities

- **Azure OpenAI**: Production-grade language models
- **Ollama**: Local development model support
- **Qdrant Cloud**: Managed vector database service
- **LangChain**: Agent orchestration and tool integration
- **Streamlit**: Modern web interface framework

## Development and Customization

### Key Components

#### ReAct Agent System

The ReAct (Reasoning and Acting) agent provides structured problem-solving through:

- **Internal Knowledge Tool**: Searches technical documentation
- **RFP History Tool**: Queries validated historical responses
- **Web Search Tool**: Contextual internet research
- **Maximum 3 iterations**: Prevents infinite reasoning loops

#### Vector Pre-calculation System

Performance optimization through:

- **Question Embedding**: Pre-computed during initial processing
- **Session Storage**: Temporary vector caching
- **Batch Processing**: Efficient multi-question handling
- **Reuse Mechanism**: Eliminates redundant calculations

#### Automatic RFP Management

Intelligent document lifecycle:

- **Sequential Numbering**: Unique identifier assignment
- **Age Tracking**: Automatic increment per submission
- **Cleanup Process**: Removal of documents older than 20 cycles
- **Metadata Preservation**: Complete audit trail maintenance

## Maintenance and Monitoring

### Regular Maintenance Tasks

- **Database Cleanup**: Weekly removal of outdated documents
- **Vector Reindexing**: Monthly optimization of search indices
- **Performance Monitoring**: Continuous query performance tracking
- **Backup Procedures**: Regular export of validated Q&A pairs

### Troubleshooting

- **Connection Issues**: Verify Qdrant URL and API key configuration
- **Performance Degradation**: Run collection inspection and cleanup
- **Memory Issues**: Monitor vector cache size and clear as needed
- **API Limits**: Check Azure OpenAI usage quotas and rate limits

## Security and Compliance

### Data Protection

- **API Key Management**: Secure environment variable storage
- **Document Isolation**: Logical separation of client data
- **Access Controls**: Role-based interface restrictions
- **Audit Trails**: Complete operation logging and traceability

### Privacy Considerations

- **Local Processing**: Option for on-premises deployment
- **Data Retention**: Configurable document lifecycle policies
- **Anonymization**: Support for sensitive information filtering
- **Export Controls**: Selective data extraction capabilities

## Contributing

### Development Standards

- **Code Quality**: PEP 8 compliance with type hints
- **Testing**: Comprehensive unit and integration tests
- **Documentation**: Detailed docstrings and API documentation
- **Version Control**: Structured commit messages and branching

### Commit Message Format

```
feat: add new functionality
fix: resolve bug or issue
docs: update documentation
refactor: code restructuring
test: add or modify tests
```

## License

MIT License - Professional enterprise software for RFP automation
