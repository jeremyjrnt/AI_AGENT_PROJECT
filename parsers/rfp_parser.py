"""
RFP Parser with Dev/Prod mode support
Dev Mode: Pattern-based extraction (free, no LLM required)
Prod Mode: Enhanced OpenAI-powered extraction with improved accuracy
"""

import os
import re
import json
import uuid
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

import pandas as pd

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import settings


class RFPParser:
    """
    RFP document parser with Dev/Prod mode support
    Dev Mode: Free pattern-based extraction using regex
    Prod Mode: Enhanced OpenAI-powered extraction with better accuracy
    """
    
    def __init__(self, mode="dev", model=None):
        """
        Initialize RFP parser with mode selection
        
        Args:
            mode: "dev" for pattern-based extraction or "prod" for LLM-enhanced extraction
            model: LLM model to use (auto-selected based on mode if None)
        """
        self.mode = mode.lower()
        
        # Auto-select model based on mode
        if model is None:
            if self.mode == "prod":
                self.model = "gpt-4o-mini"  # Fast and cost-effective for production
            else:
                self.model = "pattern"  # No LLM needed in dev mode
        else:
            self.model = model
        
        # Initialize LLM for production mode
        if self.mode == "prod":
            self._initialize_llm()
        
        print(f"📄 RFPParser initialized in {self.mode.upper()} mode")
    
    def _initialize_llm(self):
        """Initialize LLM for production mode question extraction"""
        try:
            if not settings.OPENAI_API_KEY:
                print("⚠️ OpenAI API key not found. Falling back to dev mode.")
                self.mode = "dev"
                self.model = "pattern"
                self.client = None
                return
            
            import openai
            
            self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
            print(f"🚀 Production Mode: Using OpenAI model {self.model} for question extraction")
            
        except ImportError:
            print("⚠️ openai package not installed. Falling back to dev mode.")
            self.mode = "dev"
            self.model = "pattern"
            self.client = None
        except Exception as e:
            print(f"⚠️ LLM initialization failed: {e}. Falling back to dev mode.")
            self.mode = "dev"
            self.model = "pattern"
            self.client = None
    
    def read_pdf_text(self, pdf_path: str) -> str:
        """Read PDF into plain text. Prefer PyMuPDF; fallback to PyPDF2."""
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")

        # Try PyMuPDF (fitz) first
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            pages = []
            for p in doc:
                pages.append(p.get_text())
            text = "\n".join(pages).strip()
            if text:
                return text
        except Exception:
            pass

        # Fallback: PyPDF2
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(pdf_path)
            pages = []
            for p in reader.pages:
                pages.append(p.extract_text() or "")
            text = "\n".join(pages).strip()
            return text
        except Exception as e:
            raise RuntimeError(f"PDF parse failed with both backends: {e}")
    
    def chunk_text(self, text: str, max_chunk_size: int = 4000) -> List[Dict[str, str]]:
        """Light chunking by headings and size; returns [{'content','section_ref'}, ...]."""
        heading_pattern = r'^(\d+(\.\d+)*)\s+|^(Section|Chapitre|Annex(e)?)\b|^Page\s+\d+'
        chunks, current, section = [], [], "Introduction"
        lines = text.splitlines()

        for line in lines:
            line_stripped = line.strip()
            if re.match(heading_pattern, line_stripped, re.IGNORECASE):
                if current:
                    blob = "\n".join(current).strip()
                    if len(blob) > 50:
                        chunks.append({"content": blob, "section_ref": section})
                    current = []
                section = (line_stripped[:80] or section)
            current.append(line)
            if len("\n".join(current)) > max_chunk_size:
                blob = "\n".join(current).strip()
                if len(blob) > 50:
                    chunks.append({"content": blob, "section_ref": section})
                current = []

        if current:
            blob = "\n".join(current).strip()
            if len(blob) > 50:
                chunks.append({"content": blob, "section_ref": section})
        return chunks
    
    def _extract_with_openai(self, chunk_text: str, section_ref: str) -> List[Dict]:
        """Extract questions using OpenAI API (Production mode only)."""
        if self.mode != "prod" or not hasattr(self, 'client') or self.client is None:
            return self._extract_with_patterns(chunk_text, section_ref)
        
        try:
            system_prompt = """You are an RFP question extraction expert. Extract ALL questions from the text, including:

1. EXPLICIT questions (direct questions with ? marks)
2. IMPLICIT questions (requirements, specifications, requests for information)

Focus on: security, SLAs, data residency, support, certifications, compliance, pricing, technical specs, submission requirements, deadlines, contact info.

Return ONLY a JSON array of objects with these EXACT keys:
- qid: unique identifier  
- question_text: the question/requirement
- question_type: "explicit" or "implicit"
- section_ref: section reference from text
- priority: "high", "medium", or "low" 
- deadline_hint: any deadline mentioned or null
- answer_guidance: how to answer or null
- entities: relevant entities/companies mentioned (array)
- original_span: original text excerpt (max 200 chars)
- confidence: 0.0-1.0 confidence score

Return valid JSON array only. No markdown, no explanations."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Extract questions from this RFP section ({section_ref}):\n\n{chunk_text}"}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean markdown fences if present
            if content.startswith('```'):
                lines = content.split('\n')
                content = '\n'.join(lines[1:-1])
            
            # Parse JSON
            try:
                questions = json.loads(content)
                if isinstance(questions, dict):
                    questions = [questions]
                
                # Normalize and add missing fields
                normalized = []
                for q in questions:
                    normalized.append({
                        "qid": str(uuid.uuid4())[:8],
                        "question_text": q.get("question_text", "").strip(),
                        "question_type": q.get("question_type", "implicit"),
                        "section_ref": q.get("section_ref") or section_ref,
                        "priority": q.get("priority", "medium"),
                        "deadline_hint": q.get("deadline_hint"),
                        "answer_guidance": q.get("answer_guidance"),
                        "entities": q.get("entities", []),
                        "original_span": str(q.get("original_span", ""))[:200],
                        "confidence": float(q.get("confidence", 0.9)),
                        "processing_mode": self.mode,
                        "llm_model": self.model
                    })
                
                print(f"🚀 Extracted {len(normalized)} questions from {section_ref} using OpenAI")
                return normalized
                
            except json.JSONDecodeError as e:
                print(f"⚠️ OpenAI JSON parsing failed for {section_ref}: {e}")
                return self._extract_with_patterns(chunk_text, section_ref)  # Fallback
                
        except Exception as e:
            print(f"⚠️ OpenAI API failed for {section_ref}: {e}")
            return self._extract_with_patterns(chunk_text, section_ref)  # Fallback
    
    def _extract_with_patterns(self, chunk_text: str, section_ref: str) -> List[Dict]:
        """Extract questions using regex patterns (Dev mode and fallback)."""
        try:
            questions = []
            
            # 1. Find explicit questions (with ? marks)
            explicit_questions = re.findall(r'([^.!?]*\?[^.!?]*)', chunk_text, re.IGNORECASE | re.MULTILINE)
            
            for i, q in enumerate(explicit_questions):
                q = q.strip()
                if len(q) > 10:  # Filter out very short questions
                    questions.append({
                        "qid": str(uuid.uuid4())[:8],
                        "question_text": q.replace('\n', ' ').strip(),
                        "question_type": "explicit",
                        "section_ref": section_ref,
                        "priority": "medium",
                        "deadline_hint": None,
                        "answer_guidance": None,
                        "entities": [],
                        "original_span": q[:200],
                        "confidence": 0.9,
                        "processing_mode": self.mode,
                        "llm_model": None
                    })
            
            # 2. Find implicit questions (requirements patterns)
            requirement_patterns = [
                r'(?:vendor|supplier|bidder|contractor)\s+(?:must|shall|should|will)\s+([^.!?]+[.!?])',
                r'(?:provide|submit|include|demonstrate|ensure|maintain)\s+([^.!?]+[.!?])',
                r'(?:requirements?\s+for|specification(?:s)?\s+for|details?\s+of)\s+([^.!?]+[.!?])',
                r'(?:describe|explain|detail|specify|list|identify)\s+([^.!?]+[.!?])',
                r'(?:what|how|when|where|which|why)\s+([^.!?]+[.!?])',
                r'(?:compliance with|adherence to|certification(?:s)?\s+for)\s+([^.!?]+[.!?])',
                r'(?:support|maintenance|training|documentation)\s+(?:plan|strategy|approach|method)\s+([^.!?]+[.!?])',
                r'(?:security|privacy|data protection|backup|disaster recovery)\s+([^.!?]+[.!?])',
                r'(?:pricing|cost|budget|fee|charge)\s+([^.!?]+[.!?])',
                r'(?:deadline|due date|timeline|schedule|delivery)\s+([^.!?]+[.!?])'
            ]
            
            for pattern in requirement_patterns:
                matches = re.findall(pattern, chunk_text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    requirement = match.strip()
                    if len(requirement) > 15:  # Filter out very short requirements
                        # Convert requirement to question format
                        question_text = f"What are the requirements for {requirement.lower()}"
                        
                        # Determine priority based on keywords
                        priority = "medium"
                        if any(keyword in requirement.lower() for keyword in ["security", "compliance", "certification", "mandatory", "required"]):
                            priority = "high"
                        elif any(keyword in requirement.lower() for keyword in ["optional", "preferred", "desirable"]):
                            priority = "low"
                        
                        # Check for deadline hints
                        deadline_hint = None
                        deadline_match = re.search(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d+ days?|\d+ weeks?|\d+ months?)', requirement, re.IGNORECASE)
                        if deadline_match:
                            deadline_hint = deadline_match.group(1)
                        
                        questions.append({
                            "qid": str(uuid.uuid4())[:8],
                            "question_text": question_text,
                            "question_type": "implicit",
                            "section_ref": section_ref,
                            "priority": priority,
                            "deadline_hint": deadline_hint,
                            "answer_guidance": f"Address: {requirement[:100]}...",
                            "entities": [],
                            "original_span": requirement[:200],
                            "confidence": 0.7,
                            "processing_mode": self.mode,
                            "llm_model": None
                        })
            
            # 3. Find submission/contact requirements
            submission_patterns = [
                r'(?:submit|send|deliver|provide).*?(?:to|at|via|through)\s+([^.!?]+[.!?])',
                r'(?:contact|reach|email|call)\s+([^.!?]+[.!?])',
                r'(?:proposal|bid|response).*?(?:deadline|due)\s+([^.!?]+[.!?])'
            ]
            
            for pattern in submission_patterns:
                matches = re.findall(pattern, chunk_text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    submission = match.strip()
                    if len(submission) > 10:
                        questions.append({
                            "qid": str(uuid.uuid4())[:8],
                            "question_text": f"What is the submission requirement: {submission[:50]}...",
                            "question_type": "implicit",
                            "section_ref": section_ref,
                            "priority": "high",
                            "deadline_hint": None,
                            "answer_guidance": "Follow submission guidelines carefully",
                            "entities": [],
                            "original_span": submission[:200],
                            "confidence": 0.8,
                            "processing_mode": self.mode,
                            "llm_model": None
                        })
            
            # Remove duplicates based on similar text
            unique_questions = []
            seen_texts = set()
            
            for q in questions:
                text_key = q["question_text"].lower()[:50]  # First 50 chars for comparison
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    unique_questions.append(q)
            
            mode_indicator = "🛠️" if self.mode == "dev" else "🔄"
            print(f"{mode_indicator} Extracted {len(unique_questions)} questions from {section_ref} using pattern-based approach")
            return unique_questions
            
        except Exception as e:
            print(f"⚠️ Pattern extraction failed for {section_ref}: {e}")
            # Return a basic question as fallback
            return [{
                "qid": str(uuid.uuid4())[:8],
                "question_text": f"Please review and respond to requirements in {section_ref}",
                "question_type": "implicit",
                "section_ref": section_ref,
                "priority": "medium",
                "deadline_hint": None,
                "answer_guidance": "Review the section carefully for specific requirements",
                "entities": [],
                "original_span": chunk_text[:200] if chunk_text else "",
                "confidence": 0.5,
                "processing_mode": self.mode,
                "llm_model": None
            }]
    
    def extract_questions_from_chunk(self, chunk_text: str, section_ref: str) -> List[Dict]:
        """Extract questions from a text chunk using mode-appropriate method"""
        if self.mode == "prod":
            return self._extract_with_openai(chunk_text, section_ref)
        else:
            return self._extract_with_patterns(chunk_text, section_ref)
    
    def parse_rfp_text(self, text: str) -> List[Dict]:
        """Chunk → extract per chunk → concat (no dedup)."""
        print(f"🔍 Parsing RFP text in {self.mode.upper()} mode")
        
        chunks = self.chunk_text(text)
        all_questions = []
        
        for chunk in chunks:
            try:
                questions = self.extract_questions_from_chunk(chunk["content"], chunk["section_ref"])
                all_questions.extend(questions)
            except Exception as ex:
                print(f"⚠️ Warning: chunk {chunk['section_ref']} skipped: {ex}")
        
        print(f"✅ Total: {len(all_questions)} questions extracted in {self.mode.upper()} mode")
        return all_questions
    
    def save_questions_to_excel(self, questions: List[Dict], filepath: str) -> str:
        """Write Excel with exact columns order."""
        # Create DataFrame from questions
        df = pd.DataFrame(questions)
        
        # Create output DataFrame with required columns: Questions, Answer, Comments
        df_out = pd.DataFrame({
            'Questions': df['question_text'] if 'question_text' in df.columns else '',
            'Answer': '',
            'Comments': ''
        })
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        
        # Save to Excel
        df_out.to_excel(filepath, index=False)
        print(f"💾 Questions saved to: {filepath}")
        return filepath
    
    def extract_and_export(self, text: str, excel_name: Optional[str] = None) -> Tuple[List[Dict], str]:
        """Parse and export; return (questions, excel_path)."""
        questions = self.parse_rfp_text(text)
        
        if excel_name is None:
            mode_suffix = f"_{self.mode}" if self.mode == "prod" else ""
            excel_name = f"rfp_questions{mode_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        if not excel_name.endswith(".xlsx"):
            excel_name += ".xlsx"
        
        filepath = os.path.join("outputs", excel_name)
        self.save_questions_to_excel(questions, filepath)
        
        return questions, filepath
    
    def extract_from_pdf(self, pdf_path: str, excel_name: Optional[str] = None) -> Tuple[List[Dict], str]:
        """Convenience: read PDF → extract_and_export."""
        text = self.read_pdf_text(pdf_path)
        if not text:
            raise ValueError("PDF appears empty or unreadable.")
        return self.extract_and_export(text, excel_name)


# Backward compatibility functions (using dev mode by default)
def read_pdf_text(pdf_path: str) -> str:
    """Legacy function - uses dev mode for backward compatibility"""
    parser = RFPParser(mode="dev")
    return parser.read_pdf_text(pdf_path)

def chunk_text(text: str, max_chunk_size: int = 4000) -> List[Dict[str, str]]:
    """Legacy function - uses dev mode for backward compatibility"""
    parser = RFPParser(mode="dev")
    return parser.chunk_text(text, max_chunk_size)

def call_pattern_extraction(chunk_text: str, section_ref: str) -> List[Dict]:
    """Legacy function - uses dev mode for backward compatibility"""
    parser = RFPParser(mode="dev")
    return parser._extract_with_patterns(chunk_text, section_ref)

def call_llm_json(chunk_text: str, section_ref: str, model_name: str = "pattern", openai_api_key: Optional[str] = None) -> List[Dict]:
    """Legacy function with mode detection"""
    mode = "prod" if (openai_api_key and model_name.startswith("gpt")) else "dev"
    parser = RFPParser(mode=mode, model=model_name if mode == "prod" else None)
    return parser.extract_questions_from_chunk(chunk_text, section_ref)

def parse_rfp_text(text: str, model_name: str = "pattern", openai_api_key: Optional[str] = None) -> List[Dict]:
    """Legacy function with mode detection"""
    mode = "prod" if (openai_api_key and model_name.startswith("gpt")) else "dev"
    parser = RFPParser(mode=mode, model=model_name if mode == "prod" else None)
    return parser.parse_rfp_text(text)

def save_questions_to_excel(questions: List[Dict], filepath: str) -> str:
    """Legacy function"""
    parser = RFPParser(mode="dev")
    return parser.save_questions_to_excel(questions, filepath)

def extract_and_export(text: str, excel_name: Optional[str] = None, model_name: str = "pattern", openai_api_key: Optional[str] = None) -> Tuple[List[Dict], str]:
    """Legacy function with mode detection"""
    mode = "prod" if (openai_api_key and model_name.startswith("gpt")) else "dev"
    parser = RFPParser(mode=mode, model=model_name if mode == "prod" else None)
    return parser.extract_and_export(text, excel_name)

def extract_from_pdf(pdf_path: str, excel_name: Optional[str] = None, model_name: str = "pattern", openai_api_key: Optional[str] = None) -> Tuple[List[Dict], str]:
    """Legacy function with mode detection"""
    mode = "prod" if (openai_api_key and model_name.startswith("gpt")) else "dev"
    parser = RFPParser(mode=mode, model=model_name if mode == "prod" else None)
    return parser.extract_from_pdf(pdf_path, excel_name)


# Test function for both modes
def test_rfp_parser_modes():
    """Test both dev and prod modes"""
    print("🧪 Testing RFP Parser - Both Modes")
    print("=" * 50)
    
    # Sample RFP text for testing
    sample_rfp = """
    Section 1: Introduction
    The vendor must provide a comprehensive security solution. What are your certifications?
    
    Section 2: Requirements
    Bidders shall submit their proposals by December 31, 2024.
    Provide details of your backup and disaster recovery plan.
    What is your uptime SLA guarantee?
    
    Section 3: Submission
    All proposals must be submitted to procurement@company.com by 5 PM EST.
    """
    
    print("🛠️ DEVELOPMENT MODE TEST")
    print("-" * 30)
    dev_parser = RFPParser(mode="dev")
    dev_questions = dev_parser.parse_rfp_text(sample_rfp)
    print(f"Dev mode extracted: {len(dev_questions)} questions")
    
    if settings.OPENAI_API_KEY:
        print("\n🚀 PRODUCTION MODE TEST")
        print("-" * 30)
        prod_parser = RFPParser(mode="prod")
        prod_questions = prod_parser.parse_rfp_text(sample_rfp)
        print(f"Prod mode extracted: {len(prod_questions)} questions")
    else:
        print("\n⚠️ PRODUCTION MODE SKIPPED")
        print("Set OPENAI_API_KEY in .env to test production mode")
    
    print("\n🎉 RFP Parser mode testing complete!")


if __name__ == "__main__":
    test_rfp_parser_modes()
