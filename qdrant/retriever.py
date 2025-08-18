#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qdrant Retriever
Handles retrieval and search operations from Qdrant vector database
Support for triple retrieval: DATA collection, RFP collection, and web search
"""

from typing import List, Dict, Any, Optional, Tuple
from langchain_huggingface import HuggingFaceEmbeddings
import settings
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime


class QdrantRetriever:
    """
    Handles triple retrieval operations:
    1. DATA collection (internal documentation)
    2. RFP collection (previous Q&A pairs) 
    3. Web search (DuckDuckGo for external SaaS info)
    """
    
    def __init__(self, client=None, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the retriever with Qdrant client and embedding model
        
        Args:
            client: QdrantClient instance
            embedding_model: HuggingFace model name for embeddings
        """
        self.client = client
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Collection names for triple retrieval
        self.DATA_COLLECTION = "documentation_data"
        self.RFP_COLLECTION = "rfp_qa_pairs"
        self.WEB_COLLECTION = "web_search_cache"  # Optional caching
    
    def embed_question(self, question: str) -> List[float]:
        """Convert question to embedding vector"""
        return self.embeddings.embed_query(question)
    
    def triple_retrieval(self, question: str, top_k: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """
        Perform triple retrieval for a single question
        
        Args:
            question: The RFP question to answer
            top_k: Number of results per source
            
        Returns:
            {
                'data_collection': [...],
                'rfp_collection': [...], 
                'web_search': [...]
            }
        """
        results = {
            'data_collection': [],
            'rfp_collection': [],
            'web_search': []
        }
        
        try:
            # 1. Search DATA collection (internal docs)
            results['data_collection'] = self.search_data_collection(question, top_k)
            
            # 2. Search RFP collection (previous Q&A)
            results['rfp_collection'] = self.search_rfp_collection(question, top_k)
            
            # 3. Web search via DuckDuckGo
            results['web_search'] = self.web_search_duckduckgo(question, top_k)
            
        except Exception as e:
            print(f"⚠️ Error in triple retrieval for '{question[:50]}...': {e}")
        
        return results
    
    def search_data_collection(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search internal DATA collection (documentation, technical specs)
        """
        if not self.client:
            return []
            
        try:
            query_vector = self.embed_question(query)
            
            search_results = self.client.search(
                collection_name=self.DATA_COLLECTION,
                query_vector=query_vector,
                limit=top_k
            )
            
            results = []
            for hit in search_results:
                results.append({
                    'id': hit.id,
                    'score': hit.score,
                    'content': hit.payload.get('content', ''),
                    'metadata': hit.payload.get('metadata', {}),
                    'source': 'data_collection'
                })
            
            print(f"📄 Found {len(results)} results in DATA collection")
            return results
            
        except Exception as e:
            print(f"⚠️ DATA collection search failed: {e}")
            return []
    
    def search_rfp_collection(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search RFP collection (previous question-answer pairs)
        """
        if not self.client:
            return []
            
        try:
            query_vector = self.embed_question(query)
            
            search_results = self.client.search(
                collection_name=self.RFP_COLLECTION,
                query_vector=query_vector,
                limit=top_k
            )
            
            results = []
            for hit in search_results:
                results.append({
                    'id': hit.id,
                    'score': hit.score,
                    'question': hit.payload.get('question', ''),
                    'answer': hit.payload.get('answer', ''),
                    'metadata': hit.payload.get('metadata', {}),
                    'source': 'rfp_collection'
                })
            
            print(f"🔍 Found {len(results)} similar Q&A pairs in RFP collection")
            return results
            
        except Exception as e:
            print(f"⚠️ RFP collection search failed: {e}")
            return []
    
    def web_search_duckduckgo(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search web via DuckDuckGo for external SaaS/general information
        """
        try:
            # Enhance query for SaaS/technical context
            enhanced_query = f"{query} SaaS enterprise security compliance"
            
            # Simple DuckDuckGo search (avoiding official API limits)
            search_url = f"https://duckduckgo.com/html/?q={requests.utils.quote(enhanced_query)}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(search_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                results = []
                
                # Extract search results
                result_links = soup.find_all('a', class_='result__a')[:top_k]
                
                for i, link in enumerate(result_links):
                    title = link.get_text(strip=True)
                    url = link.get('href', '')
                    
                    # Try to get snippet
                    snippet = ""
                    result_div = link.find_parent('div', class_='result__body')
                    if result_div:
                        snippet_elem = result_div.find('div', class_='result__snippet')
                        if snippet_elem:
                            snippet = snippet_elem.get_text(strip=True)
                    
                    if title and len(title) > 10:  # Filter out empty results
                        results.append({
                            'id': f'web_{i}',
                            'score': 1.0 - (i * 0.1),  # Simple relevance scoring
                            'title': title,
                            'url': url,
                            'content': snippet,
                            'source': 'web_search'
                        })
                
                print(f"🌐 Found {len(results)} web search results")
                return results
            
        except Exception as e:
            print(f"⚠️ Web search failed: {e}")
        
        return []
    
    def formulate_answer(self, question: str, retrieval_results: Dict[str, List[Dict[str, Any]]]) -> str:
        """
        Formulate an answer based on triple retrieval results
        
        Args:
            question: The original RFP question
            retrieval_results: Results from triple_retrieval()
            
        Returns:
            Formulated answer combining all sources
        """
        answer_parts = []
        
        # Process DATA collection results
        data_sources = retrieval_results.get('data_collection', [])
        if data_sources:
            answer_parts.append("📄 **Based on internal documentation:**")
            for result in data_sources[:2]:  # Top 2 results
                content = result.get('content', '')[:200]
                answer_parts.append(f"- {content}...")
        
        # Process RFP collection results  
        rfp_sources = retrieval_results.get('rfp_collection', [])
        if rfp_sources:
            answer_parts.append("\n🔍 **Similar questions answered before:**")
            for result in rfp_sources[:2]:
                question_ref = result.get('question', '')[:100]
                answer_ref = result.get('answer', '')[:150]
                answer_parts.append(f"- Q: {question_ref}...")
                answer_parts.append(f"  A: {answer_ref}...")
        
        # Process web search results
        web_sources = retrieval_results.get('web_search', [])
        if web_sources:
            answer_parts.append("\n🌐 **External SaaS industry information:**")
            for result in web_sources[:2]:
                title = result.get('title', '')
                content = result.get('content', '')[:150]
                answer_parts.append(f"- {title}: {content}...")
        
        # Combine all sources
        if answer_parts:
            return "\n".join(answer_parts)
        else:
            return f"⚠️ No relevant information found for: {question}"
    
    def save_qa_pair(self, question: str, answer: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Save validated question-answer pair to RFP collection
        
        Args:
            question: The RFP question
            answer: Human-validated answer
            metadata: Additional metadata
            
        Returns:
            True if saved successfully
        """
        if not self.client:
            print("⚠️ No Qdrant client available for saving Q&A pair")
            return False
            
        try:
            # Create embedding for the question
            question_vector = self.embed_question(question)
            
            # Prepare payload
            payload = {
                'question': question,
                'answer': answer,
                'metadata': metadata or {},
                'timestamp': datetime.now().isoformat()
            }
            
            # Generate unique ID
            import hashlib
            import uuid
            qa_text = f"{question}{answer}"
            qa_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, qa_text))
            
            # Insert into RFP collection
            self.client.upsert(
                collection_name=self.RFP_COLLECTION,
                points=[{
                    'id': qa_id,
                    'vector': question_vector,
                    'payload': payload
                }]
            )
            
            print(f"✅ Saved Q&A pair to RFP collection: {qa_id}")
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to save Q&A pair: {e}")
            return False
    
    def process_excel_questions(self, questions_df) -> List[Dict[str, Any]]:
        """
        Process all questions from Excel through triple retrieval pipeline
        
        Args:
            questions_df: DataFrame with RFP questions
            
        Returns:
            List of questions with retrieval results and formulated answers
        """
        processed_questions = []
        
        for idx, row in questions_df.iterrows():
            question = row.get('question_text', '')
            if not question or len(question.strip()) < 10:
                continue
                
            print(f"\n🔄 Processing question {idx+1}: {question[:60]}...")
            
            # Perform triple retrieval
            retrieval_results = self.triple_retrieval(question, top_k=3)
            
            # Formulate answer
            formulated_answer = self.formulate_answer(question, retrieval_results)
            
            # Prepare result
            processed_question = {
                'id': row.get('qid', f'q_{idx}'),
                'question': question,
                'section_ref': row.get('section_ref', ''),
                'priority': row.get('priority', 'medium'),
                'retrieval_results': retrieval_results,
                'formulated_answer': formulated_answer,
                'human_validated': False,
                'final_answer': None
            }
            
            processed_questions.append(processed_question)
        
        print(f"✅ Processed {len(processed_questions)} questions through triple retrieval")
        return processed_questions
    
    def human_validation_loop(self, processed_questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Interactive human validation loop for answers
        
        Args:
            processed_questions: Questions with formulated answers
            
        Returns:
            Questions with human-validated answers
        """
        validated_questions = []
        
        for i, pq in enumerate(processed_questions):
            print(f"\n{'='*60}")
            print(f"Question {i+1}/{len(processed_questions)}")
            print(f"Section: {pq['section_ref']}")
            print(f"Priority: {pq['priority']}")
            print(f"\n📝 QUESTION:")
            print(pq['question'])
            print(f"\n🤖 FORMULATED ANSWER:")
            print(pq['formulated_answer'])
            
            while True:
                print(f"\n{'='*40}")
                choice = input("👤 [A]ccept / [M]odify / [S]kip / [Q]uit: ").strip().upper()
                
                if choice == 'A':
                    # Accept the formulated answer
                    pq['final_answer'] = pq['formulated_answer']
                    pq['human_validated'] = True
                    validated_questions.append(pq)
                    print("✅ Answer accepted")
                    break
                    
                elif choice == 'M':
                    # Modify the answer
                    print("\n✏️ Enter your modified answer (press Enter twice to finish):")
                    modified_lines = []
                    while True:
                        line = input()
                        if line == "" and modified_lines and modified_lines[-1] == "":
                            break
                        modified_lines.append(line)
                    
                    modified_answer = "\n".join(modified_lines[:-1])  # Remove last empty line
                    if modified_answer.strip():
                        pq['final_answer'] = modified_answer
                        pq['human_validated'] = True
                        validated_questions.append(pq)
                        print("✅ Answer modified and saved")
                        break
                    else:
                        print("⚠️ Empty answer, try again")
                        
                elif choice == 'S':
                    # Skip this question
                    print("⏭️ Question skipped")
                    break
                    
                elif choice == 'Q':
                    # Quit validation loop
                    print(f"🛑 Validation stopped. Processed {len(validated_questions)} questions.")
                    return validated_questions
                    
                else:
                    print("⚠️ Invalid choice. Use A/M/S/Q")
        
        return validated_questions
    
    def save_validated_pairs(self, validated_questions: List[Dict[str, Any]]) -> int:
        """
        Save all validated Q&A pairs to RFP collection
        
        Args:
            validated_questions: Questions with human-validated answers
            
        Returns:
            Number of pairs saved successfully
        """
        saved_count = 0
        
        for vq in validated_questions:
            if vq.get('human_validated') and vq.get('final_answer'):
                metadata = {
                    'section_ref': vq.get('section_ref', ''),
                    'priority': vq.get('priority', 'medium'),
                    'validation_timestamp': datetime.now().isoformat()
                }
                
                if self.save_qa_pair(vq['question'], vq['final_answer'], metadata):
                    saved_count += 1
        
        print(f"✅ Saved {saved_count}/{len(validated_questions)} validated Q&A pairs")
        return saved_count
    
    # Legacy compatibility methods (simplified implementations)
    def search_similar(self, query: str, collection_name: str = None, top_k: int = 3) -> List[Dict[str, Any]]:
        """Legacy method - redirects to appropriate collection search"""
        if collection_name == self.DATA_COLLECTION:
            return self.search_data_collection(query, top_k)
        elif collection_name == self.RFP_COLLECTION:
            return self.search_rfp_collection(query, top_k)
        else:
            return self.search_data_collection(query, top_k)  # Default
    
    def search_with_filter(self, query: str, filter_conditions: Dict[str, Any], collection_name: str = None, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search with metadata filtering - basic implementation"""
        # For now, just do regular search - can be enhanced with Qdrant filters
        return self.search_similar(query, collection_name, top_k)
    
    def get_document_by_id(self, doc_id: str, collection_name: str = None) -> Optional[Dict[str, Any]]:
        """Retrieve specific document by ID"""
        if not self.client:
            return None
            
        try:
            result = self.client.retrieve(
                collection_name=collection_name or self.DATA_COLLECTION,
                ids=[doc_id]
            )
            
            if result:
                return {
                    'id': result[0].id,
                    'payload': result[0].payload
                }
        except Exception as e:
            print(f"⚠️ Document retrieval failed: {e}")
            
        return None
    
    def hybrid_search(self, query: str, keywords: List[str], collection_name: str = None, top_k: int = 3) -> List[Dict[str, Any]]:
        """Hybrid search combining semantic and keyword matching"""
        # Enhanced query with keywords
        enhanced_query = f"{query} {' '.join(keywords)}"
        return self.search_similar(enhanced_query, collection_name, top_k)