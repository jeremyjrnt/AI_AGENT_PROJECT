"""
RFP Parser using pattern-based extraction (free, no LLM required).
Reads PDF, extracts questions using regex patterns, exports to Excel.

OpenAI integration available but commented out - uncomment when needed.
"""

import os
import re
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import pandas as pd


# ---------- PDF READER ----------

def read_pdf_text(pdf_path: str) -> str:
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


# ---------- CHUNKING ----------

def chunk_text(text: str, max_chunk_size: int = 4000) -> List[Dict[str, str]]:
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


# ---------- PATTERN-BASED EXTRACTION ----------

# OPTION PAYANTE - OpenAI API (commentée pour l'instant)
# Décommentez cette fonction pour utiliser GPT-3.5-turbo à la place des patterns
"""
def call_openai_extraction(chunk_text: str, section_ref: str, api_key: str) -> List[Dict]:
    \"\"\"Extract questions using OpenAI API.\"\"\"
    try:
        import openai
        
        client = openai.OpenAI(api_key=api_key)
        
        system_prompt = \"\"\"You are an RFP question extraction expert. Extract ALL questions from the text, including:

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

Return valid JSON array only. No markdown, no explanations.\"\"\"

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract questions from this RFP section ({section_ref}):\\n\\n{chunk_text}"}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        content = response.choices[0].message.content.strip()
        
        # Clean markdown fences if present
        if content.startswith('```'):
            lines = content.split('\\n')
            content = '\\n'.join(lines[1:-1])
        
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
                    "confidence": float(q.get("confidence", 0.8))
                })
            
            print(f"✅ Extracted {len(normalized)} questions from {section_ref} using OpenAI")
            return normalized
            
        except json.JSONDecodeError as e:
            print(f"⚠️ OpenAI JSON parsing failed for {section_ref}: {e}")
            return call_pattern_extraction(chunk_text, section_ref)  # Fallback
            
    except Exception as e:
        print(f"⚠️ OpenAI API failed for {section_ref}: {e}")
        return call_pattern_extraction(chunk_text, section_ref)  # Fallback
"""


def call_pattern_extraction(chunk_text: str, section_ref: str) -> List[Dict]:
    """Extract questions using regex patterns (fallback method)."""
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
                    "confidence": 0.9
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
                        "confidence": 0.7
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
                        "confidence": 0.8
                    })
        
        # Remove duplicates based on similar text
        unique_questions = []
        seen_texts = set()
        
        for q in questions:
            text_key = q["question_text"].lower()[:50]  # First 50 chars for comparison
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_questions.append(q)
        
        print(f"✅ Extracted {len(unique_questions)} questions from {section_ref} using pattern-based approach")
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
            "confidence": 0.5
        }]


def call_llm_json(chunk_text: str, section_ref: str, model_name: str = "pattern", openai_api_key: Optional[str] = None) -> List[Dict]:
    """Main extraction function - uses pattern-based approach (OpenAI commented out)."""
    
    # Pour l'instant, on utilise seulement les patterns (gratuit)
    # Pour activer OpenAI : décommentez call_openai_extraction() et utilisez la ligne ci-dessous
    # if openai_api_key and model_name.startswith("gpt"):
    #     return call_openai_extraction(chunk_text, section_ref, openai_api_key)
    
    return call_pattern_extraction(chunk_text, section_ref)


# ---------- PUBLIC API ----------

def parse_rfp_text(text: str, model_name: str = "pattern", openai_api_key: Optional[str] = None) -> List[Dict]:
    """Chunk → extract per chunk → concat (no dedup)."""
    chunks = chunk_text(text)
    all_q = []
    for ch in chunks:
        try:
            all_q.extend(call_llm_json(ch["content"], ch["section_ref"], model_name, openai_api_key))
        except Exception as ex:
            print(f"Warning: chunk {ch['section_ref']} skipped: {ex}")
    return all_q


def save_questions_to_excel(questions: List[Dict], filepath: str) -> str:
    """Write Excel with exact columns order."""
    # Nouvelle structure : Questions, Answer, Comments
    df = pd.DataFrame(questions)
    # 'Questions' = question_text, 'Answer' et 'Comments' vides
    df_out = pd.DataFrame({
        'Questions': df['question_text'] if 'question_text' in df else None,
        'Answer': '',
        'Comments': ''
    })
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    df_out.to_excel(filepath, index=False)
    return filepath


def extract_and_export(text: str, excel_name: Optional[str] = None, model_name: str = "pattern", openai_api_key: Optional[str] = None) -> Tuple[List[Dict], str]:
    """Parse and export; return (questions, excel_path)."""
    qs = parse_rfp_text(text, model_name, openai_api_key)
    if excel_name is None:
        excel_name = f"rfp_questions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    if not excel_name.endswith(".xlsx"):
        excel_name += ".xlsx"
    path = save_questions_to_excel(qs, os.path.join("outputs", excel_name))
    return qs, path


def extract_from_pdf(pdf_path: str, excel_name: Optional[str] = None, model_name: str = "pattern", openai_api_key: Optional[str] = None) -> Tuple[List[Dict], str]:
    """Convenience: read PDF → extract_and_export."""
    text = read_pdf_text(pdf_path)
    if not text:
        raise ValueError("PDF appears empty or unreadable.")
    return extract_and_export(text, excel_name, model_name, openai_api_key)
