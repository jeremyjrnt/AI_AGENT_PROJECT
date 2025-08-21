# qdrant/

This folder contains vector database integration and retrieval logic.

- `__init__.py`: Marks this directory as a Python package.
- `cleaner.py`: Tools for cleaning up the Qdrant database.
- `client.py`: Qdrant client setup and connection logic.
- `indexer.py`: Indexing logic for RFPs and internal docs.
- `inspector.py`: Tools for inspecting the Qdrant database.
- `react_retriever.py`: ReAct-based retriever for AI-powered question answering.
- `retriever.py`: General retrieval logic for vector search. Not used in practice since react_retriever is the one used.
- `rfp_tracker.py`: Manages RFP numbering and automatic cleanup of old RFPs.
