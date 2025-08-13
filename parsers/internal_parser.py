# Comments in English only
from pathlib import Path
from typing import List, Dict, Any, Union
from bs4 import BeautifulSoup
import hashlib
import json

HTML_EXTS = {".html", ".htm"}

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

def parse_html_file(file_path: Path, target_chars: int = 1200, overlap: int = 200) -> List[Dict[str, Any]]:
    """Parse one HTML file and return chunks"""
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
        text = _normalize_text(soup.get_text(" ", strip=True))
        
        if not text:
            return []
        
        # Create chunks
        chunks = []
        doc_id = _hash(str(file_path))
        
        for i, chunk_text in enumerate(_split_text(text, target_chars, overlap)):
            chunk_id = _hash(f"{doc_id}_{i}_{chunk_text[:50]}")
            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "title": title,
                "text": chunk_text,
                "source": "internal",
                "metadata": {
                    "file_name": file_path.name,
                    "file_path": str(file_path)
                }
            })
        
        return chunks
        
    except Exception as e:
        print(f"    Erreur parsing {file_path.name}: {e}")
        return []

def find_html_files(folder_path: Path) -> List[Path]:
    """Find all HTML files recursively"""
    html_files = []
    for file_path in folder_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in HTML_EXTS:
            html_files.append(file_path)
    return html_files

def parse_folder_to_data(folder_path: Union[str, Path], target_chars: int = 1200, overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Parse all HTML files in folder and return chunks
    """
    # Clean path
    if isinstance(folder_path, str):
        folder_path = folder_path.strip().replace('"', '').replace("'", '')
    
    folder_path = Path(folder_path).resolve()
    
    # Validate
    if not folder_path.exists():
        print(f" Dossier inexistant: {folder_path}")
        return []
    
    if not folder_path.is_dir():
        print(f" Pas un dossier: {folder_path}")
        return []
    
    print(f"🔍 Recherche HTML dans: {folder_path}")
    
    # Find HTML files
    html_files = find_html_files(folder_path)
    
    if not html_files:
        print(" Aucun fichier HTML trouvé")
        return []
    
    print(f" {len(html_files)} fichier(s) HTML trouvé(s)")
    
    # Parse all files
    all_chunks = []
    for i, file_path in enumerate(html_files, 1):
        print(f" ({i}/{len(html_files)}) {file_path.name}")
        
        chunks = parse_html_file(file_path, target_chars, overlap)
        all_chunks.extend(chunks)
        
        if chunks:
            print(f"   {len(chunks)} chunks")
    
    print(f"\n Total: {len(all_chunks)} chunks extraits")
    return all_chunks

def save_parsed_data(data: List[Dict[str, Any]], output_file: str = "parsed_data.json") -> bool:
    """Save data to JSON file"""
    try:
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f" Sauvé: {output_path.absolute()}")
        return True
        
    except Exception as e:
        print(f" Erreur sauvegarde: {e}")
        return False
