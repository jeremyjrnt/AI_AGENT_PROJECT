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
    Internal document parser with Dev/Prod mode support
    Dev Mode: Basic text extraction and chunking
    Prod Mode: Enhanced processing with LLM-based text improvement
    """
    
    def __init__(self, mode="dev", model=None):
        """
        Initialize parser with mode selection
        
        Args:
            mode: "dev" for basic parsing or "prod" for LLM-enhanced parsing
            model: LLM model to use (auto-selected based on mode if None)
        """
        self.mode = mode.lower()
        
        # Auto-select model based on mode
        if model is None:
            if self.mode == "prod":
                self.model = "gpt-4o-mini"  # Fast and cost-effective for production
            else:
                self.model = None  # No LLM needed in dev mode
        else:
            self.model = model
        
        # Initialize LLM for production mode
        if self.mode == "prod":
            self._initialize_llm()
        
        print(f"📄 InternalParser initialized in {self.mode.upper()} mode")
    
    def _initialize_llm(self):
        """Initialize LLM for production mode text enhancement"""
        try:
            if not settings.OPENAI_API_KEY:
                print("⚠️ OpenAI API key not found. Falling back to dev mode.")
                self.mode = "dev"
                self.llm = None
                return
            
            from langchain_openai import ChatOpenAI
            
            self.llm = ChatOpenAI(
                model=self.model,
                temperature=0.1,
                api_key=settings.OPENAI_API_KEY
            )
            print(f"🚀 Production Mode: Using OpenAI model {self.model} for text enhancement")
            
        except ImportError:
            print("⚠️ langchain_openai not installed. Falling back to dev mode.")
            self.mode = "dev"
            self.llm = None
        except Exception as e:
            print(f"⚠️ LLM initialization failed: {e}. Falling back to dev mode.")
            self.mode = "dev"
            self.llm = None
    
    def _enhance_text_with_llm(self, raw_text: str, title: Optional[str] = None) -> Dict[str, str]:
        """
        Enhance extracted text using LLM (Production mode only)
        
        Args:
            raw_text: Raw extracted text
            title: Document title if available
            
        Returns:
            Dict with enhanced text and metadata
        """
        if self.mode != "prod" or not hasattr(self, 'llm') or self.llm is None:
            return {"enhanced_text": raw_text, "summary": "", "key_topics": ""}
        
        try:
            prompt = f"""
You are processing internal company documentation for better searchability and understanding.

Document Title: {title or "Unknown"}
Raw Text: {raw_text[:2000]}...

Tasks:
1. Clean and improve the text formatting
2. Create a brief summary (2-3 sentences)
3. Identify key topics/keywords (comma-separated)

Respond in JSON format:
{{
    "enhanced_text": "cleaned and improved text",
    "summary": "brief summary of the content",
    "key_topics": "topic1, topic2, topic3"
}}
"""
            
            response = self.llm.invoke(prompt)
            
            # Try to parse JSON response
            try:
                import json
                result = json.loads(response.content)
                return {
                    "enhanced_text": result.get("enhanced_text", raw_text),
                    "summary": result.get("summary", ""),
                    "key_topics": result.get("key_topics", "")
                }
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {"enhanced_text": raw_text, "summary": "", "key_topics": ""}
                
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
            
            # Enhance text in production mode
            if self.mode == "prod":
                enhancement_result = self._enhance_text_with_llm(raw_text, title)
                processed_text = enhancement_result["enhanced_text"]
                summary = enhancement_result["summary"]
                key_topics = enhancement_result["key_topics"]
            else:
                processed_text = raw_text
                summary = ""
                key_topics = ""
            
            # Create chunks
            chunks = []
            doc_id = _hash(str(file_path))
            
            for i, chunk_text in enumerate(_split_text(processed_text, target_chars, overlap)):
                chunk_id = _hash(f"{doc_id}_{i}_{chunk_text[:50]}")
                
                chunk_data = {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "title": title,
                    "text": chunk_text,
                    "source": "internal",
                    "processing_mode": self.mode,
                    "metadata": {
                        "file_name": file_path.name,
                        "file_path": str(file_path)
                    }
                }
                
                # Add production mode enhancements
                if self.mode == "prod":
                    chunk_data["metadata"].update({
                        "summary": summary,
                        "key_topics": key_topics,
                        "enhanced_with_llm": True,
                        "llm_model": self.model
                    })
                else:
                    chunk_data["metadata"]["enhanced_with_llm"] = False
                
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
                mode_indicator = "🚀" if self.mode == "prod" else "🛠️"
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

def parse_folder_to_data(folder_path: Union[str, Path], target_chars: int = 1200, overlap: int = 200, mode: str = "dev") -> List[Dict[str, Any]]:
    """
    Parse all HTML files in folder and return chunks
    
    Args:
        folder_path: Path to folder containing HTML files
        target_chars: Target characters per chunk
        overlap: Overlap between chunks
        mode: Processing mode ("dev" or "prod")
    """
    parser = InternalParser(mode=mode)
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


# Test function for both modes
def test_parser_modes():
    """Test both dev and prod modes"""
    print("🧪 Testing Internal Parser - Both Modes")
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
    
    if settings.OPENAI_API_KEY:
        print("\n🚀 PRODUCTION MODE TEST")
        print("-" * 30)
        prod_parser = InternalParser(mode="prod")
        prod_chunks = prod_parser.parse_folder_to_data(test_folder)
    else:
        print("\n⚠️ PRODUCTION MODE SKIPPED")
        print("Set OPENAI_API_KEY in .env to test production mode")
    
    print("\n🎉 Mode comparison test complete!")


if __name__ == "__main__":
    test_parser_modes()
