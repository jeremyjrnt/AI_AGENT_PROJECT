"""
RFP Parser for Azure OpenAI - Exhaustive Question Extraction
Converts PDF RFPs into comprehensive question lists using Azure OpenAI
"""

import os
import sys
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import settings


class RFPParser:
    """
    RFP Parser that converts PDF requirements into exhaustive question lists
    using Azure OpenAI for intelligent extraction
    """
    
    def __init__(self):
        """Initialize RFP parser with Azure OpenAI and setup directories"""
        self._initialize_azure_openai()
        self._setup_directories()
        print("🚀 RFPParser initialized with Azure OpenAI")
    
    def _setup_directories(self):
        """Setup required directories for RFP workflow"""
        self.project_root = Path(__file__).parent.parent
        
        # Define directory structure using existing data/ folders
        self.new_rfps_dir = self.project_root / "data" / "new_RFPs"
        self.past_rfps_dir = self.project_root / "data" / "past_RFPs_pdf" 
        self.parsed_rfps_dir = self.project_root / "data" / "parsed_RFPs"
        
        # Create directories if they don't exist (but they should already exist)
        for directory in [self.new_rfps_dir, self.past_rfps_dir, self.parsed_rfps_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Using existing directory structure:")
        print(f"   • new_RFPs: {self.new_rfps_dir}")
        print(f"   • past_RFPs_pdf: {self.past_rfps_dir}")
        print(f"   • parsed_RFPs: {self.parsed_rfps_dir}")
    
    def _initialize_azure_openai(self):
        """Initialize Azure OpenAI client"""
        try:
            # Check Azure OpenAI configuration
            if not all([
                settings.AZURE_OPENAI_API_KEY,
                settings.AZURE_OPENAI_ENDPOINT,
                settings.AZURE_OPENAI_CHAT_DEPLOYMENT
            ]):
                raise ValueError("Azure OpenAI configuration incomplete. Check your .env file.")
            
            from openai import AzureOpenAI
            
            self.client = AzureOpenAI(
                api_key=settings.AZURE_OPENAI_API_KEY,
                api_version=settings.AZURE_OPENAI_API_VERSION,
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
            )
            
            self.model = settings.AZURE_OPENAI_CHAT_DEPLOYMENT
            print(f"🚀 Connected to Azure OpenAI: {self.model}")
            
        except ImportError:
            raise RuntimeError("Azure OpenAI package not installed. Run: pip install openai")
        except Exception as e:
            raise RuntimeError(f"Azure OpenAI initialization failed: {e}")
    
    def read_pdf_text(self, pdf_path: str) -> str:
        """Read PDF into plain text using PyMuPDF with PyPDF2 fallback"""
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        print(f"📖 Reading PDF: {Path(pdf_path).name}")
        
        # Try PyMuPDF (fitz) first - better text extraction
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            pages = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    pages.append(f"--- Page {page_num + 1} ---\n{text}")
            
            full_text = "\n\n".join(pages)
            doc.close()
            
            if full_text.strip():
                print(f"✅ PDF read successfully with PyMuPDF ({len(full_text)} characters)")
                return full_text.strip()
                
        except ImportError:
            print("⚠️ PyMuPDF not available, trying PyPDF2...")
        except Exception as e:
            print(f"⚠️ PyMuPDF failed: {e}, trying PyPDF2...")

        # Fallback to PyPDF2
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(pdf_path)
            pages = []
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(f"--- Page {page_num + 1} ---\n{text}")
            
            full_text = "\n\n".join(pages)
            
            if full_text.strip():
                print(f"✅ PDF read successfully with PyPDF2 ({len(full_text)} characters)")
                return full_text.strip()
            else:
                raise ValueError("PDF appears to be empty or contains no extractable text")
                
        except ImportError:
            raise RuntimeError("Neither PyMuPDF nor PyPDF2 is installed. Install one: pip install PyMuPDF or pip install PyPDF2")
        except Exception as e:
            raise RuntimeError(f"PDF extraction failed with both methods: {e}")
    
    def chunk_text(self, text: str, max_chunk_size: int = 15000) -> List[str]:
        """
        Chunk text for Azure OpenAI processing
        Uses larger chunks (15k chars) since we're processing everything at once for <30 pages
        """
        # For documents under 30 pages (~90k chars), we can process in fewer, larger chunks
        if len(text) <= 90000:  # ~30 pages
            # Split into 2-3 large chunks maximum
            chunk_size = min(max_chunk_size, len(text) // 2 + 1)
        else:
            chunk_size = max_chunk_size
        
        chunks = []
        current_pos = 0
        
        while current_pos < len(text):
            chunk_end = current_pos + chunk_size
            
            # If not the last chunk, try to break at a natural boundary
            if chunk_end < len(text):
                # Look for section breaks, paragraph breaks, or sentence endings
                breakpoints = [
                    text.rfind('\n\nSection', current_pos, chunk_end),
                    text.rfind('\n\n', current_pos, chunk_end),
                    text.rfind('. ', current_pos, chunk_end),
                ]
                
                # Use the best breakpoint found
                best_break = max([bp for bp in breakpoints if bp > current_pos], default=chunk_end)
                chunk_end = best_break
            
            chunk = text[current_pos:chunk_end].strip()
            if chunk:
                chunks.append(chunk)
            
            current_pos = chunk_end
        
        print(f"📑 Text chunked into {len(chunks)} parts")
        return chunks
    
    def extract_questions_from_text(self, text: str) -> List[str]:
        """
        Extract exhaustive questions from RFP text using Azure OpenAI
        """
        system_prompt = """You are an RFP question extraction expert.  
Your task is to read the following RFP text and extract ALL the questions, requirements, and expectations, whether explicit or implicit.  

Rules and guidance:
- Convert every requirement, condition, or obligation into a QUESTION form.  
  Example: "The contractor shall provide 24/7 support" → "Will the contractor provide 24/7 support?"  
- Include **implicit expectations**:  
  - If the text mentions insurance requirements → ask a question about whether the contractor maintains those insurance coverages.  
  - If the text specifies contract duration → ask a question like "Will the contractor commit to a 3-year contract term with two 1-year extension options?"  
  - If the text prohibits substitutions → ask "Will the contractor refrain from providing substitutions or equivalents?"  
- Be EXHAUSTIVE: Do not skip any requirement, clause, or condition, no matter how minor.  
- Cover ALL categories evoked in this RFP.  

- Phrase each item as a **standalone clear question** starting with "Will the contractor…", "Does the contractor…", "Can the contractor…", or "Is the contractor required to…".  
- Output format:  
  1. A numbered list of ALL extracted questions, one per line.  
  2. No summaries, no grouping — just the full question set.  

IMPORTANT: Do not miss any requirement. Even minor or administrative details must be turned into a question."""

        try:
            print("🔍 Sending text to Azure OpenAI for question extraction...")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Extract ALL questions from this RFP text:\n\n{text}"}
                ],
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=4000   # Generous token limit for comprehensive extraction
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse the numbered list into individual questions
            questions = []
            lines = content.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Remove numbering (1., 2., etc.) and clean up
                import re
                clean_line = re.sub(r'^\d+\.\s*', '', line).strip()
                
                if clean_line and len(clean_line) > 10:  # Filter very short lines
                    questions.append(clean_line)
            
            print(f"✅ Extracted {len(questions)} questions from text chunk")
            return questions
            
        except Exception as e:
            print(f"❌ Azure OpenAI extraction failed: {e}")
            return []
    
    def process_rfp_pdf(self, pdf_path: str) -> List[str]:
        """
        Complete pipeline: PDF → Text → Chunks → Questions
        """
        print(f"🔄 Processing RFP: {Path(pdf_path).name}")
        
        # Step 1: Extract text from PDF
        full_text = self.read_pdf_text(pdf_path)
        
        # Step 2: Determine processing strategy based on document size
        char_count = len(full_text)
        estimated_pages = char_count / 3000  # ~3000 chars per page
        
        all_questions = []
        
        if estimated_pages <= 30:
            print(f"📄 Small RFP (~{estimated_pages:.1f} pages) - Processing with larger chunks")
            chunks = self.chunk_text(full_text, max_chunk_size=15000)
        else:
            print(f"📚 Large RFP (~{estimated_pages:.1f} pages) - Using standard chunking")
            chunks = self.chunk_text(full_text, max_chunk_size=10000)
        
        # Step 3: Extract questions from each chunk
        for i, chunk in enumerate(chunks, 1):
            print(f"🔍 Processing chunk {i}/{len(chunks)}")
            chunk_questions = self.extract_questions_from_text(chunk)
            all_questions.extend(chunk_questions)
        
        # Step 4: Remove duplicates while preserving order
        seen = set()
        unique_questions = []
        
        for question in all_questions:
            # Create a normalized version for comparison (lowercase, no extra spaces)
            normalized = ' '.join(question.lower().split())
            if normalized not in seen and len(normalized) > 20:  # Filter very short questions
                seen.add(normalized)
                unique_questions.append(question)
        
        print(f"🎯 Final result: {len(unique_questions)} unique questions extracted")
        print(f"   (Removed {len(all_questions) - len(unique_questions)} duplicates)")
        
        return unique_questions
    
    def move_pdf_to_processed(self, pdf_path: str) -> str:
        """
        Ensure PDF ends up in data/past_RFPs_pdf after processing
        - If PDF was in new_RFPs: MOVE it to past_RFPs_pdf (remove from new_RFPs)  
        - If PDF was elsewhere: COPY it to past_RFPs_pdf (keep original)
        
        Args:
            pdf_path: Current path of the PDF file
            
        Returns:
            str: Path of the PDF in past_RFPs_pdf
        """
        pdf_path = Path(pdf_path)
        
        # Determine destination path in past_RFPs_pdf
        destination = self.past_rfps_dir / pdf_path.name
        
        # Handle filename conflicts in destination
        counter = 1
        while destination.exists():
            stem = pdf_path.stem
            suffix = pdf_path.suffix
            destination = self.past_rfps_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        
        # Check if PDF is currently in new_RFPs (using absolute paths for comparison)
        pdf_absolute = pdf_path.resolve()
        new_rfps_absolute = self.new_rfps_dir.resolve()
        
        if pdf_absolute.parent == new_rfps_absolute:
            # PDF is in new_RFPs → MOVE it (remove from new_RFPs)
            shutil.move(str(pdf_path), str(destination))
            print(f"📦 Moved PDF: {pdf_path.name} → data/past_RFPs_pdf/ (removed from new_RFPs)")
        else:
            # PDF is elsewhere → COPY it (keep original)
            shutil.copy2(str(pdf_path), str(destination))
            print(f"📄 Copied PDF: {pdf_path.name} → data/past_RFPs_pdf/ (original kept at {pdf_path.parent.name})")
        
        return str(destination)
    
    def create_excel_output(self, questions: List[str], pdf_name: str) -> str:
        """
        Create Excel file with questions in data/parsed_RFPs directory
        
        Args:
            questions: List of extracted questions
            pdf_name: Original PDF filename for naming the Excel file
            
        Returns:
            str: Path to created Excel file
        """
        # Create DataFrame with required structure
        df = pd.DataFrame({
            'Question': questions,
            'Answer': [''] * len(questions),  # Empty column
            'Comments': [''] * len(questions)  # Empty column
        })
        
        # Generate Excel filename based on PDF name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pdf_stem = Path(pdf_name).stem
        excel_filename = f"RFP_Questions_{pdf_stem}_{timestamp}.xlsx"
        excel_path = self.parsed_rfps_dir / excel_filename
        
        # Save to Excel
        df.to_excel(excel_path, index=False)
        
        print(f"💾 Excel created: {excel_filename}")
        print(f"📍 Location: data/parsed_RFPs/")
        print(f"📊 Contains {len(questions)} questions ready for answers")
        
        return str(excel_path)
    
    def parse_pdf_to_excel(self, pdf_path: str, manage_files: bool = False) -> tuple[str, str]:
        """
        Complete pipeline: PDF → Questions → Excel with optional file management
        
        Args:
            pdf_path: Path to input PDF file
            manage_files: If True, move PDF and save Excel. If False, just return paths without file operations
            
        Returns:
            tuple[str, str]: (original_pdf_path or moved_pdf_path, excel_path or temp_excel_path)
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        print(f"\n🚀 Starting RFP processing pipeline")
        print(f"📥 Input PDF: {pdf_path.name}")
        
        # Extract questions from PDF
        questions = self.process_rfp_pdf(str(pdf_path))
        
        if not questions:
            raise ValueError("No questions could be extracted from the PDF")
        
        if manage_files:
            # Create Excel output in data/parsed_RFPs directory
            excel_path = self.create_excel_output(questions, pdf_path.name)
            
            # Move PDF to data/past_RFPs_pdf directory (if it was in data/new_RFPs)
            moved_pdf_path = self.move_pdf_to_processed(str(pdf_path))
            
            print(f"\n🎉 RFP processing pipeline complete!")
            print(f"📦 PDF moved to: data/past_RFPs_pdf/{Path(moved_pdf_path).name}")
            print(f"📤 Excel created: data/parsed_RFPs/{Path(excel_path).name}")
            print(f"❓ Questions extracted: {len(questions)}")
            
            return moved_pdf_path, excel_path
        else:
            # Just extract questions without file management
            print(f"\n🎉 RFP question extraction complete!")
            print(f"📄 PDF analyzed: {pdf_path.name}")
            print(f"❓ Questions extracted: {len(questions)}")
            print("📝 No files moved or created (manage_files=False)")
            
            # Return the questions as a list and original PDF path
            return str(pdf_path), questions


# Convenience function for direct usage
def extract_rfp_questions(pdf_path: str, manage_files: bool = False) -> tuple[str, any]:
    """
    Convenience function to extract questions from RFP PDF with optional file management
    
    Args:
        pdf_path: Path to RFP PDF file
        manage_files: If True, move files and create Excel. If False, just extract questions
        
    Returns:
        tuple[str, any]: If manage_files=True: (moved_pdf_path, excel_path)
                        If manage_files=False: (original_pdf_path, questions_list)
    """
    parser = RFPParser()
    return parser.parse_pdf_to_excel(pdf_path, manage_files=manage_files)


# Test function
def test_rfp_parser():
    """Test function for RFP parser"""
    print("🧪 Testing RFP Parser")
    print("=" * 50)
    
    # Sample RFP text for testing
    sample_rfp = """
    1. INTRODUCTION
    The City of San Francisco requires a comprehensive IT support solution.
    
    2. REQUIREMENTS
    2.1 The contractor shall provide 24/7 technical support.
    2.2 All support staff must be certified in relevant technologies.
    2.3 Response time for critical issues must not exceed 2 hours.
    2.4 The contractor must maintain liability insurance of $1 million minimum.
    
    3. CONTRACT TERMS
    The initial contract term is 3 years with two 1-year extension options.
    No substitutions or equivalent products will be accepted.
    
    4. SUBMISSION REQUIREMENTS
    All proposals must be submitted by December 31, 2024, at 5:00 PM PST.
    Late submissions will not be accepted.
    """
    
    # Test question extraction
    parser = RFPParser()
    questions = parser.extract_questions_from_text(sample_rfp)
    
    print(f"✅ Extracted {len(questions)} questions from sample text:")
    for i, question in enumerate(questions, 1):
        print(f"  {i}. {question}")
    
    print("\n🎉 RFP Parser test complete!")


def test_rfp_workflow():
    """Test complete RFP workflow with file management"""
    print("🧪 Testing Complete RFP Workflow")
    print("=" * 60)
    
    # Create a sample PDF file in data/new_RFPs for testing
    parser = RFPParser()
    
    # Create sample PDF content as text file (for demonstration)
    sample_pdf_path = parser.new_rfps_dir / "sample_rfp_test.txt"
    sample_content = """
    SAMPLE RFP - CITY OF SAN FRANCISCO
    
    1. TECHNICAL REQUIREMENTS
    - The contractor shall provide 24/7 monitoring services
    - All systems must have 99.9% uptime guarantee  
    - Response time for critical alerts must be under 15 minutes
    
    2. STAFFING REQUIREMENTS  
    - All technicians must hold relevant certifications
    - Minimum 5 years experience required for senior staff
    - On-site presence required during business hours
    
    3. INSURANCE AND LIABILITY
    - General liability insurance minimum $2 million
    - Professional liability coverage required
    - Certificate of insurance must be provided
    """
    
    # Write sample file
    with open(sample_pdf_path, 'w', encoding='utf-8') as f:
        f.write(sample_content)
    
    print(f"📄 Created sample file: {sample_pdf_path.name}")
    print(f"📍 Location: data/new_RFPs/")
    
    # Test question extraction from text
    questions = parser.extract_questions_from_text(sample_content)
    
    if questions:
        print(f"\n✅ Extracted {len(questions)} questions:")
        for i, question in enumerate(questions, 1):
            print(f"  {i}. {question}")
        
        # Test Excel creation
        excel_path = parser.create_excel_output(questions, sample_pdf_path.name)
        
        # Test file movement
        moved_path = parser.move_pdf_to_processed(str(sample_pdf_path))
        
        print(f"\n🎉 Workflow test complete!")
        print(f"📦 File moved: {Path(moved_path).name}")
        print(f"📊 Excel created: {Path(excel_path).name}")
        
        return moved_path, excel_path
    else:
        print("❌ No questions extracted - test failed")
        return None, None


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "workflow":
        test_rfp_workflow()
    else:
        test_rfp_parser()
