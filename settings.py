#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Settings and configuration management for AI Agent Project
Loads environment variables from .env file
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# Qdrant Database Settings
QDRANT_URL = os.getenv('QDRANT_URL', 'https://c9c63dad-8e5c-49a3-90f5-fb71773836fa.europe-west3-0.gcp.cloud.qdrant.io')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY', None)

# Note: Collection names are defined in qdrant/client.py
# INTERNAL_COLLECTION = "internal_knowledge_base" 
# RFP_QA_COLLECTION = "rfp_qa_history"

# Dynamic Embedding Provider Settings
EMBEDDING_PROVIDER = os.getenv('EMBEDDING_PROVIDER', 'huggingface')  # "huggingface" or "openai"

# HuggingFace Embedding Settings (default/free)
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')

# OpenAI Embedding Settings (premium)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', None)
OPENAI_EMBEDDING_MODEL = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-large')

# Vector Database Settings
VECTOR_SIZE = 384  # Default for all-MiniLM-L6-v2, will be auto-detected
DISTANCE_METRIC = 'cosine'

# Dynamic vector size based on provider
def get_default_vector_size() -> int:
    """
    Get default vector size based on embedding provider
    
    Returns:
        int: Expected vector dimension
    """
    if EMBEDDING_PROVIDER == 'openai':
        if OPENAI_EMBEDDING_MODEL == 'text-embedding-3-large':
            return 3072
        elif OPENAI_EMBEDDING_MODEL == 'text-embedding-3-small':
            return 1536
        else:
            return 1536  # Default OpenAI size
    else:  # huggingface
        if 'all-MiniLM-L6-v2' in EMBEDDING_MODEL:
            return 384
        elif 'all-mpnet-base-v2' in EMBEDDING_MODEL:
            return 768
        else:
            return 384  # Default HuggingFace size

def is_openai_provider() -> bool:
    """Check if using OpenAI as embedding provider"""
    return EMBEDDING_PROVIDER.lower() == 'openai'

def is_huggingface_provider() -> bool:
    """Check if using HuggingFace as embedding provider"""
    return EMBEDDING_PROVIDER.lower() == 'huggingface'

# Retrieval Settings
DEFAULT_TOP_K = 5
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# Project Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
PARSED_DATA_DIR = DATA_DIR / 'parsed'
RAW_DATA_DIR = DATA_DIR / 'raw'

def get_settings_info():
    """
    Returns current settings configuration
    """
    return {
        'qdrant_url': QDRANT_URL,
        'collections': ['internal_knowledge_base', 'rfp_qa_history'],
        'embedding_provider': EMBEDDING_PROVIDER,
        'embedding_model': EMBEDDING_MODEL if is_huggingface_provider() else OPENAI_EMBEDDING_MODEL,
        'vector_size': get_default_vector_size(),
        'distance_metric': DISTANCE_METRIC,
        'default_top_k': DEFAULT_TOP_K,
        'chunk_size': CHUNK_SIZE,
        'chunk_overlap': CHUNK_OVERLAP,
        'has_openai_key': bool(OPENAI_API_KEY) if is_openai_provider() else False
    }
