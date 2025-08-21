#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Internal Parser Agent
Specialized agent for internal document parsing and processing
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from parsers.internal_parser import InternalParser


class InternalParserAgent:
    """Specialized agent for internal document parsing and text processing"""
    
    def __init__(self):
        self.parser = InternalParser()
        self.agent_name = "InternalParserAgent"
        self.description = "Specialized agent for parsing and processing internal documents with LLM enhancement"
    
    def parse_document(self, file_path, use_reformulation=True, chunk_size=2000, overlap=200):
        """
        Parse and process a document with intelligent text enhancement
        
        Args:
            file_path (str): Path to document
            use_reformulation (bool): Enable LLM-based text reformulation
            chunk_size (int): Size of text chunks for processing
            overlap (int): Overlap between chunks for context preservation
            
        Returns:
            list: Intelligently parsed and reformulated text chunks
        """
        return self.parser.parse_document(
            file_path=file_path,
            use_reformulation=use_reformulation,
            chunk_size=chunk_size,
            overlap=overlap
        )
    
    def parse_directory(self, directory_path, use_reformulation=True, chunk_size=2000, overlap=200):
        """
        Parse and process all documents in a directory with batch intelligence
        
        Args:
            directory_path (str): Path to directory containing documents
            use_reformulation (bool): Enable LLM-based text reformulation
            chunk_size (int): Size of text chunks for processing
            overlap (int): Overlap between chunks for context preservation
            
        Returns:
            dict: Dictionary mapping file paths to processed text chunks
        """
        return self.parser.parse_directory(
            directory_path=directory_path,
            use_reformulation=use_reformulation,
            chunk_size=chunk_size,
            overlap=overlap
        )
    
    def get_status(self):
        """Return agent status"""
        return {
            "agent_name": self.agent_name,
            "description": self.description,
            "parser_available": self.parser is not None
        }


# Test and direct usage
if __name__ == "__main__":
    agent = InternalParserAgent()
    print(f"Agent {agent.agent_name} initialized")
    print(f"Description: {agent.description}")
    print(f"Status: {agent.get_status()}")
