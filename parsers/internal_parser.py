# Comments in English only
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
from bs4 import BeautifulSoup
import hashlib
import json
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import settings

HTML_EXTS = {".html", ".htm"}

class InternalParser:
    """
    Internal document parser with Dev/Prod/Hybrid mode support
    Dev Mode: Basic text extraction and chunking
    Prod Mode: Enhanced processing with LLM-based text improvement (replaces original)
    Hybrid Mode: Keeps original text + adds LLM enhancements in metadata
    """
    
    def __init__(self, mode="dev", model=None):
        """
        Initialize parser with mode selection
        
        Args:
            mode: "dev" for basic parsing, "prod" for LLM-enhanced parsing, 
                  "hybrid" for original text + LLM metadata
            model: LLM model to use (auto-selected based on mode if None)
        """
        self.mode = mode.lower()
        
        # Auto-select model based on mode
        if model is None:
            if self.mode == "prod":
                self.model = "gpt-4o-mini"  # Fast and cost-effective for production
            elif self.mode == "hybrid":
                self.model = "qwen2.5:0.5b"  # Default Qwen model for hybrid mode (ultra-light and fast)
            else:
                self.model = None  # No LLM needed in dev mode
        else:
            self.model = model
        
        # Initialize LLM for production or hybrid modes
        if self.mode in ["prod", "hybrid"]:
            self._initialize_llm()
        
        print(f"📄 InternalParser initialized in {self.mode.upper()} mode")
    
    def _initialize_llm(self):
        """Initialize LLM for production or hybrid mode text enhancement"""
        try:
            # For hybrid mode with Qwen, use Ollama
            if self.mode == "hybrid" and "qwen" in self.model.lower():
                try:
                    from langchain_ollama import ChatOllama
                    
                    self.llm = ChatOllama(
                        model=self.model,
                        temperature=0.1,
                        base_url="http://localhost:11434"  # Default Ollama URL
                    )
                    print(f"🤖 Hybrid Mode: Using Ollama model {self.model} for text enhancement")
                    return
                except ImportError:
                    print("⚠️ langchain_ollama not installed. Install with: pip install langchain-ollama")
                    self.mode = "dev"
                    self.llm = None
                    return
            
            # For production mode or OpenAI models, use OpenAI
            if not settings.OPENAI_API_KEY:
                print(f"⚠️ OpenAI API key not found. Falling back to dev mode.")
                self.mode = "dev"
                self.llm = None
                return
            
            from langchain_openai import ChatOpenAI
            
            self.llm = ChatOpenAI(
                model=self.model,
                temperature=0.1,
                api_key=settings.OPENAI_API_KEY
            )
            print(f"🚀 {self.mode.title()} Mode: Using OpenAI model {self.model} for text enhancement")
            
        except ImportError:
            print("⚠️ Required LangChain packages not installed. Falling back to dev mode.")
            self.mode = "dev"
            self.llm = None
        except Exception as e:
            print(f"⚠️ LLM initialization failed: {e}. Falling back to dev mode.")
            self.mode = "dev"
            self.llm = None
    
    def _enhance_text_with_llm(self, raw_text: str, title: Optional[str] = None) -> Dict[str, str]:
        """
        Enhance extracted text using LLM (Production/Hybrid mode only)
        
        Args:
            raw_text: Raw extracted text
            title: Document title if available
            
        Returns:
            Dict with enhanced text and metadata
        """
        if self.mode not in ["prod", "hybrid"] or not hasattr(self, 'llm') or self.llm is None:
            return {"enhanced_text": raw_text, "summary": "", "key_topics": ""}
        
        try:
            # Use consistent English prompt for all modes with emphasis on longer enhanced text
            prompt = f"""
You are an expert document processor improving internal company documentation for better searchability.

Document Title: {title or "Unknown"}
Raw Text Content: {raw_text[:2000]}...

INSTRUCTIONS:
1. ENHANCED_TEXT: Create a comprehensive, well-structured version of the original text. This should be the LONGEST field with 6-10 sentences minimum. Include all key details, technical information, and context from the original text. Improve readability while preserving all important information.

2. SUMMARY: Create a brief executive summary in 2-3 sentences maximum.

3. KEY_TOPICS: List main topics and keywords (comma-separated).

CRITICAL: The enhanced_text MUST be significantly longer than the summary. Aim for 300-500+ characters for enhanced_text.

Respond in JSON format:
{{
    "enhanced_text": "comprehensive detailed text with all key information, technical details, context, and improved structure - minimum 6-10 sentences covering all aspects of the original content",
    "summary": "brief executive summary maximum 2-3 sentences",
    "key_topics": "topic1, topic2, topic3"
}}
"""
            
            response = self.llm.invoke(prompt)
            
            # Try to parse JSON response
            try:
                import json
                
                # Clean the response content (remove markdown code blocks if present)
                content = response.content.strip()
                if content.startswith("```json"):
                    content = content[7:]  # Remove ```json
                if content.endswith("```"):
                    content = content[:-3]  # Remove closing ```
                content = content.strip()
                
                # Try to extract JSON from mixed content
                if not content.startswith('{'):
                    # Look for JSON pattern in the response
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        content = json_match.group()
                    else:
                        # No JSON found, treat as plain text response
                        raise json.JSONDecodeError("No JSON pattern found", content, 0)
                
                result = json.loads(content)
                
                # Extract and validate fields
                enhanced_text = result.get("enhanced_text", "")
                summary = result.get("summary", "")
                key_topics = result.get("key_topics", "")
                
                # Check if fields seem to be swapped or contain template text
                if ("minimum 6-10 sentences" in enhanced_text or 
                    "comprehensive detailed text" in enhanced_text or
                    len(enhanced_text) < 50):
                    # Fields might be swapped or malformed, try to fix
                    if len(summary) > len(enhanced_text) and "minimum" not in summary:
                        # Swap them
                        enhanced_text, summary = summary, enhanced_text
                
                # Final validation and fallback
                if len(enhanced_text) < 50 or "minimum 6-10 sentences" in enhanced_text:
                    # Create a proper enhanced text from raw_text
                    sentences = raw_text.split('. ')
                    if len(sentences) >= 3:
                        enhanced_text = '. '.join(sentences[:6]) + '.'  # Take first 6 sentences
                    else:
                        enhanced_text = raw_text[:400] + "..."  # Take first 400 chars
                
                # Ensure summary is shorter than enhanced_text
                if len(summary) >= len(enhanced_text):
                    # Create a proper summary
                    words = enhanced_text.split()
                    if len(words) > 20:
                        summary = ' '.join(words[:20]) + "..."
                    else:
                        summary = enhanced_text[:100] + "..."
                
                return {
                    "enhanced_text": enhanced_text,
                    "summary": summary,
                    "key_topics": key_topics
                }
                
            except json.JSONDecodeError as e:
                # Fallback: treat response as plain text and create structured output
                print(f"⚠️ JSON parsing failed, using plain text fallback: {str(e)[:50]}...")
                
                # Use the raw response as enhanced text if it's substantial
                raw_response = response.content.strip()
                
                # Clean up common patterns from failed responses
                if "Enhanced Text:" in raw_response:
                    # Try to extract structured info from plain text response
                    lines = raw_response.split('\n')
                    enhanced_text = ""
                    summary = ""
                    key_topics = ""
                    
                    current_section = None
                    for line in lines:
                        line = line.strip()
                        if line.startswith("Enhanced Text:") or "enhanced_text" in line.lower():
                            current_section = "enhanced"
                            enhanced_text += line.replace("Enhanced Text:", "").strip() + " "
                        elif line.startswith("Summary:") or "summary" in line.lower():
                            current_section = "summary"
                            summary += line.replace("Summary:", "").strip() + " "
                        elif line.startswith("Key Topics:") or "key_topics" in line.lower():
                            current_section = "topics"
                            key_topics += line.replace("Key Topics:", "").strip() + " "
                        elif current_section == "enhanced" and line:
                            enhanced_text += line + " "
                        elif current_section == "summary" and line:
                            summary += line + " "
                        elif current_section == "topics" and line:
                            key_topics += line + " "
                    
                    enhanced_text = enhanced_text.strip()
                    summary = summary.strip()
                    key_topics = key_topics.strip()
                else:
                    # Use raw response as enhanced text
                    enhanced_text = raw_response[:800] if len(raw_response) > 800 else raw_response
                
                # Generate proper enhanced text from raw_text if response is poor
                if len(enhanced_text) < 100:
                    sentences = raw_text.split('. ')
                    if len(sentences) >= 4:
                        enhanced_text = '. '.join(sentences[:8]) + '.'  # Take first 8 sentences
                    else:
                        enhanced_text = raw_text[:600] + "..."  # Take first 600 chars
                
                # Create shorter summary if needed
                if not summary or len(summary) >= len(enhanced_text):
                    summary_words = enhanced_text.split()[:20]
                    summary = ' '.join(summary_words) + "..."
                
                # Create key topics if missing
                if not key_topics:
                    # Extract key words from title and enhanced text
                    import re
                    words = re.findall(r'\b[A-Z][a-z]+\b', (title or "") + " " + enhanced_text)
                    key_topics = ", ".join(list(dict.fromkeys(words))[:5])  # Remove duplicates, take first 5
                
                return {
                    "enhanced_text": enhanced_text,
                    "summary": summary,
                    "key_topics": key_topics
                }
                
        except Exception as e:
            print(f"⚠️ LLM enhancement failed: {e}")
            return {"enhanced_text": raw_text, "summary": "", "key_topics": ""}
    
    def parse_html_file(self, file_path: Path, target_chars: int = 1200, overlap: int = 200) -> List[Dict[str, Any]]:
        """Parse one HTML file and return chunks with mode-appropriate processing"""
        try:
            html_content = file_path.read_text(encoding="utf-8", errors="ignore")
            soup = BeautifulSoup(html_content, "lxml")
            
            # Extract title
            title = None
            if soup.title and soup.title.string:
                title = _normalize_text(soup.title.string)
            elif soup.find("h1"):
                title = _normalize_text(soup.find("h1").get_text())
            
            # Extract all text
            raw_text = _normalize_text(soup.get_text(" ", strip=True))
            
            if not raw_text:
                return []
            
            # Enhance text based on mode
            if self.mode in ["prod", "hybrid"]:
                enhancement_result = self._enhance_text_with_llm(raw_text, title)
                enhanced_text = enhancement_result["enhanced_text"]
                summary = enhancement_result["summary"]
                key_topics = enhancement_result["key_topics"]
            else:
                enhanced_text = ""
                summary = ""
                key_topics = ""
            
            # Choose text for chunking based on mode
            if self.mode == "prod":
                # Production mode: use enhanced text for chunking
                text_for_chunking = enhanced_text if enhanced_text else raw_text
            else:
                # Dev and Hybrid modes: always use original text for chunking
                text_for_chunking = raw_text
            
            # Create chunks
            chunks = []
            doc_id = _hash(str(file_path))
            
            for i, chunk_text in enumerate(_split_text(text_for_chunking, target_chars, overlap)):
                chunk_id = _hash(f"{doc_id}_{i}_{chunk_text[:50]}")
                
                chunk_data = {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "title": title,
                    "text": chunk_text,
                    "source": "internal",
                    "processing_mode": self.mode,
                    "metadata": {
                        "file": {
                            "name": file_path.name,
                            "path": str(file_path)
                        }
                    }
                }
                
                # Add mode-specific enhancements
                if self.mode == "prod":
                    chunk_data["metadata"]["llm"] = {
                        "model": self.model,
                        "enhanced_with_llm": True,
                        "summary": summary,
                        "key_topics": key_topics
                    }
                elif self.mode == "hybrid":
                    chunk_data["metadata"]["llm"] = {
                        "model": self.model,
                        "enhanced_with_llm": True,
                        "enhanced_text": enhanced_text,  # Full enhanced text (not chunked)
                        "summary": summary,
                        "key_topics": key_topics,
                        "is_hybrid_mode": True
                    }
                
                chunks.append(chunk_data)
            
            return chunks
            
        except Exception as e:
            print(f"    Erreur parsing {file_path.name}: {e}")
            return []
    
    def parse_folder_to_data(self, folder_path: Union[str, Path], target_chars: int = 1200, overlap: int = 200) -> List[Dict[str, Any]]:
        """
        Parse all HTML files in folder and return chunks with mode-appropriate processing
        """
        # Clean path
        if isinstance(folder_path, str):
            folder_path = folder_path.strip().replace('"', '').replace("'", '')
        
        folder_path = Path(folder_path).resolve()
        
        # Validate
        if not folder_path.exists():
            print(f"❌ Dossier inexistant: {folder_path}")
            return []
        
        if not folder_path.is_dir():
            print(f"❌ Pas un dossier: {folder_path}")
            return []
        
        print(f"🔍 Recherche HTML dans: {folder_path}")
        print(f"📄 Mode de traitement: {self.mode.upper()}")
        
        # Find HTML files
        html_files = find_html_files(folder_path)
        
        if not html_files:
            print("⚠️ Aucun fichier HTML trouvé")
            return []
        
        print(f"📁 {len(html_files)} fichier(s) HTML trouvé(s)")
        
        # Parse all files
        all_chunks = []
        for i, file_path in enumerate(html_files, 1):
            print(f"📄 ({i}/{len(html_files)}) {file_path.name}")
            
            chunks = self.parse_html_file(file_path, target_chars, overlap)
            all_chunks.extend(chunks)
            
            if chunks:
                mode_indicator = "🤖" if self.mode == "hybrid" else "🚀" if self.mode == "prod" else "🛠️"
                print(f"   {mode_indicator} {len(chunks)} chunks ({self.mode} mode)")
        
        print(f"\n✅ Total: {len(all_chunks)} chunks extraits en mode {self.mode.upper()}")
        return all_chunks


# Utility functions
def _hash(s: str) -> str:
    """Generate short hash for IDs"""
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()[:16]

def _normalize_text(text: str) -> str:
    """Clean and normalize text"""
    return " ".join(text.split())

def _split_text(text: str, target_chars: int = 1200, overlap: int = 200) -> List[str]:
    """Split text into chunks with overlap"""
    text = text.strip()
    if not text:
        return []
    
    chunks = []
    i, n = 0, len(text)
    while i < n:
        j = min(i + target_chars, n)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks

def find_html_files(folder_path: Path) -> List[Path]:
    """Find all HTML files recursively"""
    html_files = []
    for file_path in folder_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in HTML_EXTS:
            html_files.append(file_path)
    return html_files


# Backward compatibility functions (using dev mode by default)
def parse_html_file(file_path: Path, target_chars: int = 1200, overlap: int = 200) -> List[Dict[str, Any]]:
    """Legacy function - uses dev mode for backward compatibility"""
    parser = InternalParser(mode="dev")
    return parser.parse_html_file(file_path, target_chars, overlap)

def parse_folder_to_data(folder_path: Union[str, Path], target_chars: int = 1200, overlap: int = 200, mode: str = "dev", model: str = None) -> List[Dict[str, Any]]:
    """
    Parse all HTML files in folder and return chunks
    
    Args:
        folder_path: Path to folder containing HTML files
        target_chars: Target characters per chunk
        overlap: Overlap between chunks
        mode: Processing mode ("dev", "prod", or "hybrid")
        model: Specific model to use (optional)
    """
    parser = InternalParser(mode=mode, model=model)
    return parser.parse_folder_to_data(folder_path, target_chars, overlap)

def save_parsed_data(data: List[Dict[str, Any]], output_file: str = "parsed_data.json") -> bool:
    """Save data to JSON file"""
    try:
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Sauvé: {output_path.absolute()}")
        return True
        
    except Exception as e:
        print(f"❌ Erreur sauvegarde: {e}")
        return False


# Test function for all modes
def test_parser_modes():
    """Test dev, prod, and hybrid modes"""
    print("🧪 Testing Internal Parser - All Modes")
    print("=" * 50)
    
    # Test with a sample folder (you'll need to adjust the path)
    test_folder = Path("data/internal_docs")  # Adjust path as needed
    
    if not test_folder.exists():
        print(f"⚠️ Test folder {test_folder} not found")
        return
    
    print("🛠️ DEVELOPMENT MODE TEST")
    print("-" * 30)
    dev_parser = InternalParser(mode="dev")
    dev_chunks = dev_parser.parse_folder_to_data(test_folder)
    
    print("\n🤖 HYBRID MODE TEST (Qwen + Ollama)")
    print("-" * 30)
    hybrid_parser = InternalParser(mode="hybrid", model="qwen2.5:0.5b")
    hybrid_chunks = hybrid_parser.parse_folder_to_data(test_folder)
    
    if settings.OPENAI_API_KEY:
        print("\n🚀 PRODUCTION MODE TEST")
        print("-" * 30)
        prod_parser = InternalParser(mode="prod")
        prod_chunks = prod_parser.parse_folder_to_data(test_folder)
    else:
        print("\n⚠️ PRODUCTION MODE SKIPPED")
        print("Set OPENAI_API_KEY in .env to test production mode")
    
    print("\n🎉 Mode comparison test complete!")
    print("📊 Results Summary:")
    print(f"   Dev chunks: {len(dev_chunks) if 'dev_chunks' in locals() else 0}")
    print(f"   Hybrid chunks: {len(hybrid_chunks) if 'hybrid_chunks' in locals() else 0}")
    print(f"   Prod chunks: {len(prod_chunks) if 'prod_chunks' in locals() else 0}")


if __name__ == "__main__":
    test_parser_modes()
