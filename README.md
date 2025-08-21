# RFPilot - AI Agent RFP Management System

Request for Proposal (RFP) automation system leveraging advanced AI, vector databases, and ReAct agents to compose proposal response workflows.

## Why RFPilot?

Responding to Requests for Proposal (RFPs) is a critical but often painful process for organizations. Traditional RFP management is:

- **Time-consuming**: Teams spend hours manually parsing PDFs, searching past proposals, and formatting answers.  
- **Error-prone**: Copy-pasting from previous documents introduces inconsistencies and risks.  
- **Fragmented**: Knowledge is scattered across emails, spreadsheets, and archived files.  
- **Reactive**: Valuable past work is underutilized because it isn’t systematically indexed or searchable.  

**RFPilot** solves these challenges by combining advanced AI with structured workflows:

- Automatically extracts and numbers questions from incoming RFPs.  
- Uses semantic search and AI agents to draft accurate responses from internal docs, historical Q&A, and web context.  
- Ensures reliability with a **human-in-the-loop validation step**.  
- Exports professional, ready-to-submit RFP responses in Excel or PDF.  

By reducing manual effort and surfacing the best institutional knowledge, RFPilot enables organizations to **respond faster, with higher quality, and at greater scale.**

## Overview

This system provides intelligent automation for RFP processing through semantic search, automated question answering, and comprehensive document management. Built with modern AI technologies including Azure OpenAI, Qdrant vector database, and LangChain ReAct agents.

### Context

For this project, we used **Okta** as the reference company.  
Okta is a leading **Identity and Access Management (IAM)** platform that provides secure Single Sign-On (SSO), Multi-Factor Authentication (MFA), lifecycle management, and identity governance solutions for enterprises.  

We answered RFPs **as if we were Okta**, leveraging Okta’s public technical documentation to generate responses.  
The RFPs themselves were collected from **publicly available, open-source RFP documents**, primarily from **public institutions in the USA**.  
This allowed us to simulate realistic proposal workflows while staying within open-access data sources.

## Core Features

### RFP Processing Pipeline

- **Intelligent Document Parsing**: Automated extraction of questions from Excel and PDF documents
- **Sequential Numbering System**: Automatic RFP numbering with age-based cleanup mechanisms
- **Human Validation Workflow**: Structured validation process with complete audit trails
- **Automated Export**: Generation of comprehensive Excel reports with completed responses

### AI-Powered Question Answering

- **ReAct Agent Architecture**: Reasoning and Acting agents with structured tool usage
- **Triple Retrieval System**: Simultaneous search across three knowledge sources:
  - Internal knowledge base documentation with Qdrant semantic search
  - Historical Q&A validated responses using Qdrant semantic search
  - Contextual web search integration with DuckDuckSearchGoSearchRun
- **Human Validation**: A company employee validates Question/Answer/Comments trio by operating changes, adding rows and then submits the completed and validated RFP Response in an Excel format.

### Vector Database Management

- **Qdrant Integration**: High-performance vector similarity search
- **Multiple Collections**: Logical separation of knowledge domains
- **Semantic Search**: Cosine similarity on 1536D vectors
- **Metadata Enrichment**: Complete traceability with timestamps, validators, and RFP identifiers
- **Automatic Cleanup**: Configurable age-based document removal system

### Professional User Interface

- **Corporate Design**: Clean, professional interface suitable for enterprise environments
- **Modular Architecture**: Separated workflow and documentation interfaces
- **Advanced Styling**: Custom CSS with hover effects and professional color schemes
- **Responsive Layout**: Adaptive design with optimized sidebar and wide layouts

### Management Tools

- **Analytics Dashboard**: Real-time metrics and vector database statistics
- **Collection Inspector**: Advanced debugging and data visualization tools
- **Maintenance Suite**: Reset, cleanup, and optimization utilities

## Agent Overview and Detailed Functionality

The `agent/` directory contains modular AI agents, each responsible for a specific aspect of the RFP automation workflow. These agents encapsulate advanced logic for parsing, completion, and orchestration, and interact with the rest of the system through well-defined interfaces.

### 1. RFPParserAgent

**Purpose:**  
Extracts structured questions from raw RFP PDF documents.

**How it works:**  
- Receives a PDF file (uploaded or selected from the UI).
- Uses advanced parsing logic (leveraging `parsers/rfp_parser.py`) to analyze the document structure, identify question blocks, and extract them into a structured format.
- Outputs a list of questions, which are then displayed in the UI for review and editing.
- Handles edge cases such as multi-line questions, tables, and inconsistent formatting in source PDFs.

**Role in the workflow:**  
Serves as the entry point for transforming unstructured RFPs into a machine-readable, editable format.

---

### 2. RFPCompletionAgent

**Purpose:**  
Automatically completes RFP answers using AI, orchestrates batch processing, and manages the end-to-end RFP completion workflow.

**Parsing steps:**
1. **File Reading:**  
   Each document is read and split into logical sections or chunks, typically based on headings, paragraphs, or semantic boundaries.

2. **Text Cleaning:**  
   Raw text is cleaned to remove boilerplate, navigation, and formatting artifacts, ensuring only meaningful content is retained.

3. **Enhanced Text Generation (Qwen2.5-0.5B):**  
   For each chunk, the system uses the Qwen2.5-0.5B language model to generate:
   - **Enhanced Text:**  
     The original chunk is rewritten or expanded for clarity, completeness, and context. This step enriches the content, making it more informative and semantically dense.
   - **Summary:**  
     A concise summary of the chunk is generated, capturing the main idea in a few sentences.
   - **Key Topics:**  
     The model extracts a list of key topics or concepts covered in the chunk.

   These fields are stored alongside the original text in the metadata.

4. **Embedding Creation:**  
   **Instead of embedding the raw original text, the system embeds the Enhanced Text.**  
   This means the vector representation in Qdrant is based on the richer, contextually improved version of the content.

### Why Use Enhanced Text for Embedding?

- **Semantic Density:**  
  Enhanced text contains more context, explanations, and clarifications, making the embedding more representative of the true meaning in a concise way.
- **Improved Retrieval:**  
  When users or agents search for answers, the semantic search is more likely to find relevant and high-quality matches, even if the query uses different wording than the original document.
- **Robustness:**  
  The enhanced text removes HTML tags or noise. Moreover, summaries and key topics in the metadata allow for advanced filtering, topic-based search, and explainability in results.

---

### 3. InternalParserAgent

**Purpose:**  
Indexes and parses internal documentation to enrich the knowledge base used for RFP completion.

**How it works:**  
- Processes internal documents (such as policies, technical docs, or compliance files) in HTML formats.
- Extracts relevant content and indexes it into the vector database (`qdrant`) for semantic retrieval.
- Supports incremental updates and re-indexing as new documents are added.

**Role in the workflow:**  
Provides the foundational knowledge required for accurate and context-aware RFP answer generation, enabling the system to leverage organizational expertise.

---

## Agent Collaboration

- **RFPParserAgent** extracts questions → **RFPCompletionAgent** generates answers (using context from **InternalParserAgent**-indexed docs and other sources).
- All agents interact with the vector database (Qdrant) for storage, retrieval, and semantic search.
- The UI orchestrates agent calls based on user actions (upload, parse, pre-complete, submit).

---

**Summary Table:**

| Agent                | Main Function                | Key Methods/Features                | Interacts With           |
|----------------------|------------------------------|-------------------------------------|--------------------------|
| RFPParserAgent       | Extract questions from PDF    | `parse_rfp_file`, `parse_pdf_to_excel` | parsers/rfp_parser.py    |
| RFPCompletionAgent   | AI answer completion & submit | `process_excel_file`, `complete_rfp`   | qdrant/react_retriever.py, Qdrant |
| InternalParserAgent  | Index internal docs           | `parse_internal_docs

## System Specifications

### Vector Database Collections content

| Collection Name             | Document Count | Vector Dimension | Primary Usage                            |
| --------------------------- | -------------- | ---------------- | ---------------------------------------- |
| `internal_knowledge_base` | 1,348          | 1536D            | Technical Documentation from the Company |
| `rfp_qa_history`          | 284            | 1536D            | Validated Q&A responses from Past RFPs   |

- **`internal_knowledge_base`**  
  This collection contains embeddings of the **documentation and information extracted from the Okta website**.  
  For the purpose of this project, we submitted only **modest documentation** about Okta solutions — not exhaustive coverage — but it was **sufficient to test the pipeline** and demonstrate **promising results**.  

- **`rfp_qa_history`**  
  This collection contains embeddings from **4 RFPs**:  
  1. **One real RFP** that Okta had actually filled (very uncommon to find).  
  2. **Two synthetic RFPs**, generated with AI under our supervision, designed to introduce diversity with more classical and common RFP questions.  
  3. **One real RFP** that exists publicly, which we **completed with AI assistance and human supervision**.  

⚠️ **Note**: This dataset is **very limited** compared to a real-world deployment, where a company would typically have:  
- Dozens or hundreds of past RFPs embedded in `rfp_qa_history`.  
- Richer and higher-quality internal data in `internal_knowledge_base` (beyond parsed website data).  

This project demonstrates the **feasibility of the pipeline** with minimal data, but in production the approach would scale with much larger and higher-quality documentation.

### Vector Database MetaData

`internal_knowledge_base` Collection :

Each entry (vector) in the `internal_knowledge_base` collection represents a chunk of internal documentation. The metadata attached to each vector provides rich context for semantic search, filtering, and explainability.

**Typical metadata fields:**

- **original_text**: The raw text extracted from the document chunk.
- **enhanced_text**: The improved, context-rich version of the original text (generated by Qwen2.5-0.5B), used for embedding.
- **summary**: A concise summary of the chunk’s content.
- **key_topics**: List of key topics or concepts covered in the chunk.
- **source_file**: The filename or path of the original document.
- **section**: The section or heading within the document where the chunk was found.
- **embedded_field**: Always set to `"enhanced_text"` to indicate which field was embedded.
- **indexed_at**: Timestamp of when the chunk was indexed.
- **document_type**: Typically `"internal_doc_chunk"`.

**Example:**
```json
{
  "original_text": "Okta provides SSO and MFA for enterprise applications...",
  "enhanced_text": "Okta delivers secure Single Sign-On (SSO) and Multi-Factor Authentication (MFA), enabling organizations to protect access to all enterprise applications with a unified identity platform.",
  "summary": "Okta offers unified SSO and MFA to secure enterprise app access.",
  "key_topics": ["SSO", "MFA", "Identity Management", "Enterprise Security"],
  "source_file": "Okta_Security_Guide.pdf",
  "section": "Authentication Overview",
  "embedded_field": "enhanced_text",
  "indexed_at": "2025-08-21T14:32:10",
  "document_type": "internal_doc_chunk"
}
```

rfp_qa_history Collection

The `rfp_qa_history` collection stores validated **Question/Answer/Comments** triplets from completed RFPs.  
Each entry is enriched with metadata to ensure **traceability, auditability, and advanced search/filtering** capabilities.

### Typical Metadata Fields

- **question**: The RFP question (used for embedding).  
- **answer**: The validated answer provided.  
- **comments**: Additional comments or context for the answer.  
- **rfp_name**: The name or identifier of the RFP.  
- **rfp_file**: The filename of the original RFP document.  
- **rfp_path**: The path to the RFP file.  
- **rfp_age**: Sequential number (age) of the RFP, used for cleanup logic.  
- **submitter_name**: The person who submitted the completed RFP.  
- **question_index**: The index of the question within the RFP.  
- **indexed_at**: Timestamp of when the Q&A was indexed.  
- **source**: The origin of the RFP (e.g., organization, public dataset, synthetic).  
- **source_type**: Typically `"completed_rfp"`.  
- **document_type**: Always `"rfp_qa_pair"`.  
- **embedded_field**: Always `"question"`, indicating which field was embedded.  

### Example Entry

```json
{
  "question": "Does your solution support SAML-based SSO?",
  "answer": "Yes, Okta fully supports SAML 2.0 for Single Sign-On.",
  "comments": "See Okta documentation for supported integrations.",
  "rfp_name": "RFP_SanDiego",
  "rfp_file": "RFP_SanDiego_excel.xlsx",
  "rfp_path": "data/completed_RFPs_excel/RFP_SanDiego_excel.xlsx",
  "rfp_age": 17,
  "submitter_name": "Jessica Wilson",
  "validator_name": "Paul Johnson",
  "question_index": 3,
  "indexed_at": "2025-08-21T15:12:44",
  "source": "San Diego County",
  "source_type": "completed_rfp",
  "document_type": "rfp_qa_pair",
  "embedded_field": "question"
}
```


## Integration Capabilities

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

#### Automatic RFP Management

Intelligent document lifecycle:

- **Age Tracking**: Automatic increment per submission
- **Cleanup Process**: Removal of documents older than 20 cycles
- **Metadata Preservation**: Complete audit trail maintenance

Example of Execution :

A full log of this execution is provided in the repository under [`ReAct_log_example`](./ReAct_log_example.txt).  
It shows how the agent reasons step by step, queries the vector databases, corrects its own tool usage, and ultimately produces a final draft answer.  

### Why ReAct is Useful in RFP Answering

This example demonstrates how the ReAct agent mimics a **human reasoning process**:

1. **Starts with historical knowledge**  
   The agent first searches the `rfp_qa_history` collection to check if a similar question has already been answered in past RFPs.  
   This ensures **consistency** across responses and reuses validated knowledge.  

2. **Expands to internal documentation**  
   When past RFPs did not fully address the question even if it found the closest questions, the agent switched to the `internal_knowledge_base`.  
   This ensures alignment with **company policies and documentation** and auto-critic.  

3. **Synthesizes an answer**  
   By choosing the most relevant source of information, the agent generates a **final draft response** that aligns with standard RFP evaluation criteria and internal practices.  

4. **Knows when not to answer**  
   Especially here, the ReAct finds enough information to give a ruled out answer but when it is not the case, it returns 'Please review this section' or similar.,Unlike naive systems that simply select the *closest semantic match* and risk producing misleading outputs, ReAct can recognize when it does **not have enough information** to confidently answer.  
   In such cases, it can **escalate to a human reviewer** instead of fabricating an answer.  
   This capability:  
   - Builds **trust** with users by avoiding low-quality, hallucinated responses.  
   - Prevents **time-consuming reviews** of incorrect AI answers.  
   - Makes the agent **more valuable**, since a system that can admit uncertainty is far more reliable in practice.  



## Current Handling of Input Formats

RFPilot is designed to be flexible with document formats, but in the current implementation the workflow focuses on two primary input types:

- **RFPs as PDF files**  
  In practice, nearly all Requests for Proposal are received as PDFs. The system automatically parses these files to extract questions and structure them for processing.

- **Internal Documentation as HTML files**  
  For internal knowledge, documentation was downloaded in bulk (e.g., Okta technical site) and stored as a folder of HTML files. These are parsed, indexed, and embedded into the vector database for semantic search and retrieval.

## User Interface

The RFPilot system provides a **professional and user-friendly interface** to interact with the pipeline.

- Everything in the interface is clearly explained, and there is also a dedicated **"User Guide" window** to help new users navigate the workflow.  
- The `data/new_RFPs/` folder contains sample RFPs that can be tested directly in the interface.  
  These are the same inputs we used to generate the examples shown in the [`examples/`](./examples) folder.  
- In the sidebar, the **Knowledge Base Management** section is currently **disabled**, since the uploader was not fully functional at the time of submission.  
  This feature is planned for future versions, but it is **not required** to process or reproduce results because the internal knowledge base (`internal_knowledge_base` collection with 1,348 points) has already been embedded for this project.  

Overall, the interface allows end-to-end testing of the RFP workflow without additional configuration.

**NOTE :** In this project, a simple natural language prompt is obviously not adequate to trigger the agent’s workflow. Instead, the **selection of an RFP document** itself acts as the prompt that initiates the processing flow.

---

## RFP Age and History Management

To ensure the quality and relevance of AI-generated answers, the system implements an **RFP age** mechanism for managing the history of completed RFPs.

### What is RFP Age?

Each time a new RFP is submitted and indexed, it is assigned a sequential **age** (an incrementing number). This age represents the order in which RFPs have been processed and helps track the recency of each RFP in the system.

### Why Keep Only the 20 Most Recent RFPs?

- **Relevance:**  
  Requirements and best practices evolve over time. By keeping only the 20 most recent RFPs in the `rfp_qa_history` collection, the system ensures that the AI (especially the ReAct agent) relies on up-to-date answers and avoids outdated or deprecated information.
- **Avoid Contradictions:**  
  If too many old RFPs are kept, the AI may retrieve and consider conflicting answers for the same question, leading to confusion or incorrect completions. Limiting the history reduces the risk of surfacing contradictory information.
- **Optimal Balance:**  
  The number 20 is chosen as a balance: it is large enough to provide a rich base of recent, validated answers, but small enough to minimize the risk of outdated or conflicting data.

### How is This Implemented?

- After each successful RFP submission, the system automatically checks the age of all RFPs in the `rfp_qa_history` collection.
- If there are more than 20, the oldest RFPs (by age) are deleted, keeping only the 20 most recent.
- This cleanup is handled transparently and ensures that the vector database remains focused on the most relevant and current knowledge.


## Repository Documentation

Each main folder in the project contains its own **README.md** file.  
These sub-readmes explain the **role of each file** in that folder, helping contributors and users quickly understand the purpose of every component without needing to trace the entire codebase.

## Examples Folder

The `examples/` directory provides sample input and output files to help users understand the end-to-end workflow of the system.

- `examples/input_examples/` contains example RFP PDF files (e.g., `1.pdf`, `2.pdf`, `3.pdf`) that can be used as test inputs for the parsing and completion pipeline.
- `examples/output_examples/` contains the corresponding processed Excel files (e.g., `1.xlsx`, `2.xlsx`, `3.xlsx`) generated by the system after running the input PDFs through the full RFP automation workflow.

**Correspondence:**  
Each file with the same number in both folders represents the same RFP processed through the system.  
For example:
- `input_examples/1.pdf` → `output_examples/1.xlsx`
- `input_examples/2.pdf` → `output_examples/2.xlsx`
- `input_examples/3.pdf` → `output_examples/3.xlsx`

This structure allows you to easily compare the original RFP input with the final, AI-completed

### Source of Input Examples

The input example PDFs are also available in the `data/new_RFPs/` folder:  

- **Input 1 – `RFP_synthetic1`**  
  A synthetic RFP PDF with **classic questions**.  
  Purpose: demonstrate a simple and basic test case for the pipeline.  

- **Input 2 – `RFP_SERS`**  
  A **12-page** RFP containing **implicit questions** embedded in text.  
  Purpose: represent a **medium-difficulty test case**, requiring more semantic parsing.  

- **Input 3 – `RFP_SanMateo`**  
  A **28-page** RFP with **table visuals and implicit questions**.  
  Purpose: illustrate a **hard test case**, where parsing and interpretation are more challenging.  

This structure allows to cover scenarios from **easy → medium → hard**.


## Technical Architecture

```
AI_AGENT_PROJECT/
├── agent/                # AI agent logic for parsing and completion
│   ├── rfp_parser_agent.py        # Extracts questions from RFP PDFs
│   ├── rfp_completion_agent.py    # AI-powered answer completion
│   └── internal_parser_agent.py   # Parses internal documentation
├── parsers/              # Document parsing utilities
│   ├── rfp_parser.py              # Core RFP PDF/Excel parser
│   └── internal_parser.py         # Internal doc parser
├── qdrant/                # Vector database integration and management
│   ├── client.py                  # Qdrant client setup
│   ├── indexer.py                 # Indexing logic for RFPs/docs
│   ├── retriever.py               # Vector search and retrieval
│   ├── react_retriever.py         # ReAct agent orchestration
│   ├── rfp_tracker.py             # RFP numbering and cleanup
│   ├── inspector.py               # Collection inspection tools
│   └── cleaner.py                 # Maintenance and cleanup scripts
├── ui/                    # User interface
│   └── rfp_manager.py             # Streamlit web app for RFP management
├── data/                  # All input, output, and archived documents
│   └── new_RFPs/                  # Incoming RFP PDFs to process

├── outputs/               # Generated Excel files and reports
├── tokens_count/          # Token usage tracking and reporting
│   ├── usage_tracker.py           # Persistent usage summary
│   └── ...                        # Token logs and reports
├── test/                  # Unit and integration tests
├── settings.py            # Centralized configuration
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
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
```

## Questions and Support

If you have any questions about using RFPilot, or if you would like guidance on reproducing our results, please feel free to reach out to us, we will be happy to help.

## Future Work in Short-Term & Improvements

While RFPilot already demonstrates promising results, there are several avenues for improvement and extension:

1. **Improved Internal Data Parsing with Larger LLMs**  
   Use larger, more capable language models to parse internal documentation.  
   This would generate richer `enhanced_text` representations, improving the **quality of the embedded knowledge base** and resulting in more accurate answers.

2. **Per-Question Re-Completion**  
   Introduce the ability to **re-run the completion pipeline for a specific question** when the initial result is not satisfactory, instead of regenerating the entire RFP response.  
   This would give users more flexibility and fine-grained control over the final output.

3. **Extended Data Format Support**  
   Expand beyond current formats to support:  
   - **Internal data**: not only HTML, but also Markdown, DOCX, PDF, and other enterprise documentation formats.  
   - **RFP inputs**: in addition to PDFs, support Word, Excel, and structured online submission formats.  
   This would broaden the applicability of the system to more real-world use cases.

4. **Answer Justification & Traceability**  
   Leverage detailed **ReAct execution logs** to provide the justification for each answer, citing which sources and reasoning steps were used.  
   This would deliver **greater transparency and interpretability** to the user, fostering trust in the system by showing *why* a particular answer was chosen.

5. **Knowledge Base Management Uploader in UI**  
   Make fully functional the **Knowledge Base Management Uploader** in the user interface.  
   Currently, the corresponding code exists but was commented out since it was not fully working and was not critical for testing or reproducing results.  
   For this project, the `internal_knowledge_base` collection was already pre-embedded with **1,348 points** (see [Vector Database Collections](#vector-database-collections)).  
   Completing this feature would allow users to upload and manage their own internal documentation collections directly through the UI.
