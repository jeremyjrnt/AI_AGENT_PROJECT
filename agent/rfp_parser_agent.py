"""
RFP Parser Agent: Advanced AI agent for intelligent RFP document analysis and question extraction.

This sophisticated agent specializes in understanding RFP documents, extracting meaningful questions,
and applying advanced parsing strategies to transform unstructured RFP content into actionable data.
"""

from typing import List, Dict, Any
from parsers.rfp_parser import RFPParser


class RFPParserAgent:
    """
    Advanced AI agent for sophisticated RFP document analysis.
    
    This agent employs intelligent parsing algorithms to analyze RFP documents,
    extract questions, and understand document structure with AI-enhanced capabilities.
    """
    
    def __init__(self):
        """Initialize the RFP Parser Agent with advanced parsing capabilities."""
        self.rfp_parser = RFPParser()
        self.description = "AI-powered RFP document analysis and question extraction agent"
    
    def parse_rfp_questions(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Intelligently parse RFP documents to extract questions with AI-enhanced analysis.
        
        Args:
            file_path (str): Path to the RFP document
            
        Returns:
            List[Dict[str, Any]]: Extracted questions with metadata
        """
        return self.rfp_parser.parse_rfp_questions(file_path)
    
    def parse_rfp_file(self, file_path: str) -> tuple:
        """
        Parse RFP file and return questions list with Excel output path.
        
        This method provides the main interface for UI integration, processing
        RFP documents and generating both question lists and Excel outputs.
        
        Args:
            file_path (str): Path to the RFP document
            
        Returns:
            tuple: (questions_list, excel_path)
        """
        # First extract questions
        questions = self.rfp_parser.process_rfp_pdf(file_path)
        
        if not questions:
            raise ValueError("No questions could be extracted from the PDF")
        
        # Create Excel output and move files
        from pathlib import Path
        pdf_path = Path(file_path)
        excel_path = self.rfp_parser.create_excel_output(questions, pdf_path.name)
        
        # Move PDF to processed directory
        self.rfp_parser.move_pdf_to_processed(file_path)
        
        return questions, excel_path
    
    def get_rfp_structure(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze and understand the structure of RFP documents using advanced AI techniques.
        
        Args:
            file_path (str): Path to the RFP document
            
        Returns:
            Dict[str, Any]: Structured representation of the RFP document
        """
        # Implement intelligent structure analysis
        questions = self.parse_rfp_questions(file_path)
        return {
            "total_questions": len(questions),
            "questions": questions,
            "document_path": file_path,
            "analysis_type": "AI-enhanced RFP parsing"
        }
