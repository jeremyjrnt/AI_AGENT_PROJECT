#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RFP Completion Agent
Master orchestration agent for end-to-end RFP processing and intelligent completion
"""

import sys
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from .rfp_parser_agent import RFPParserAgent
from qdrant.react_retriever import ReactRFPRetriever


class RFPCompletionAgent:
    """Master orchestration agent coordinating intelligent RFP processing workflows"""
    
    def __init__(self, mode="prod"):
        self.rfp_parser = RFPParserAgent()
        self.react_retriever = ReactRFPRetriever(mode=mode)
        self.agent_name = "RFPCompletionAgent"
        self.description = "Master agent orchestrating end-to-end intelligent RFP processing and completion workflows"
    
    def complete_rfp(self, rfp_file_path, collection_name="internal_knowledge_base"):
        """
        Execute comprehensive end-to-end RFP processing with intelligent analysis
        
        Args:
            rfp_file_path (str): Path to RFP document requiring processing
            collection_name (str): Knowledge base collection for intelligent responses
            
        Returns:
            dict: Comprehensive processing results with intelligent insights
        """
        # Step 1: Parse RFP and extract questions
        questions, excel_path = self.rfp_parser.parse_rfp_file(rfp_file_path)
        
        # Step 2: Process Excel file with batch ReAct retrieval
        processed_excel_path, question_vectors = self.react_retriever.process_rfp_excel_batch(
            excel_path, 
            batch_size=3
        )
        
        return {
            "total_questions": len(questions),
            "excel_path": processed_excel_path,
            "question_vectors": len(question_vectors),
            "status": "completed"
        }
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a single question using ReAct methodology.
        
        This method provides a simple interface for the UI to get answers
        for individual questions using the ReAct retrieval system.
        
        Args:
            question (str): Question to answer
            
        Returns:
            Dict: Answer result with all necessary fields for UI
        """
        return self.react_retriever.answer_rfp_question(question)
    
    def process_excel_file(self, excel_path: str, batch_size: int = 3, output_path: str = None) -> tuple[str, List[Dict]]:
        """
        Process an Excel file with RFP questions using batch processing.
        
        This method wraps the process_rfp_excel_batch function from ReactRFPRetriever
        to provide a clean interface for batch processing RFP questions.
        
        Args:
            excel_path (str): Path to the Excel file with questions
            batch_size (int): Number of questions to process in each batch
            output_path (str, optional): Path for output file
            
        Returns:
            tuple: (processed_excel_path, question_vectors_list)
        """
        return self.react_retriever.process_rfp_excel_batch(
            excel_path=excel_path,
            batch_size=batch_size,
            output_path=output_path
        )
    
    def get_status(self):
        """Return comprehensive agent status and orchestration capabilities"""
        return {
            "agent_name": self.agent_name,
            "description": self.description,
            "orchestration_components": {
                "rfp_parser_agent": self.rfp_parser is not None,
                "react_rfp_retriever": self.react_retriever is not None
            },
            "capabilities": [
                "End-to-end RFP orchestration",
                "Batch Excel processing", 
                "ReAct-based question answering",
                "Comprehensive result synthesis"
            ]
        }


# Test and direct usage
if __name__ == "__main__":
    agent = RFPCompletionAgent()
    print(f"Agent {agent.agent_name} initialized")
    print(f"Description: {agent.description}")
    print(f"Status: {agent.get_status()}")
    print("Ready to process complete RFPs!")
