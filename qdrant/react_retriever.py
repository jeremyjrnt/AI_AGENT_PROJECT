"""
ReAct-based Retriever for RFP Pre-completion
Intelligently manages three information sources using reasoning and action cycles
Supports both Development (Free Ollama) and Production (OpenAI) modes
"""

import os
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# LangChain imports for ReAct
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.tools import DuckDuckGoSearchRun

# Project imports
from qdrant.client import get_qdrant_client
import settings

class ReactRFPRetriever:
    """
    ReAct-based retriever with Dev/Prod mode support
    Dev Mode: Uses free Ollama models locally
    Prod Mode: Uses OpenAI models with API key
    """
    
    def __init__(self, client=None, mode="dev", model=None):
        """
        Initialize the ReAct RFP Retriever with mode selection
        
        Args:
            client: Qdrant client instance
            mode: "dev" for Ollama (free) or "prod" for OpenAI (paid)
            model: Model to use (auto-selected based on mode if None)
        """
        self.client = client or get_qdrant_client()
        self.mode = mode.lower()
        self.DATA_COLLECTION = "internal_knowledge_base"
        self.RFP_COLLECTION = "rfp_qa_history"
        
        # Auto-select model based on mode
        if model is None:
            if self.mode == "prod":
                self.model = settings.AZURE_OPENAI_CHAT_DEPLOYMENT or "team11-gpt4o"  # Use Azure deployment name
            else:
                self.model = "qwen2:0.5b"   # Free local model for development
        else:
            self.model = model
        
        # Initialize LLM based on mode
        self._initialize_llm()
        
        # Initialize embeddings (for vector search)
        self._initialize_embeddings()
        
        # Initialize tools and agent
        self._setup_tools()
        self._setup_agent()
    
    def _initialize_llm(self):
        """Initialize LLM based on development or production mode"""
        if self.mode == "prod":
            # Production mode: Use Azure OpenAI
            try:
                from langchain_openai import AzureChatOpenAI
                
                if not settings.AZURE_OPENAI_API_KEY:
                    raise ValueError("Azure OpenAI API key not found. Set AZURE_OPENAI_API_KEY in .env file for production mode.")
                
                self.llm = AzureChatOpenAI(
                    azure_deployment=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
                    api_version=settings.AZURE_OPENAI_API_VERSION,
                    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                    api_key=settings.AZURE_OPENAI_API_KEY,
                    temperature=0.1,
                    model_name=settings.AZURE_OPENAI_CHAT_DEPLOYMENT
                )
                print(f"🚀 Production Mode: Using Azure OpenAI model {settings.AZURE_OPENAI_CHAT_DEPLOYMENT}")
                
            except ImportError:
                raise ImportError("langchain_openai not installed. Run: pip install langchain-openai")
            except Exception as e:
                raise Exception(f"Failed to initialize Azure OpenAI: {e}")
        
        else:
            # Development mode: Use Ollama (free)
            try:
                from langchain_ollama import ChatOllama
                
                self.llm = ChatOllama(
                    model=self.model,
                    temperature=0.1,
                    base_url="http://localhost:11434"
                )
                print(f"🛠️ Development Mode: Using Ollama model {self.model}")
                
            except ImportError:
                raise ImportError("langchain_ollama not installed. Run: pip install langchain-ollama")
            except Exception as e:
                print(f"⚠️ Ollama initialization failed: {e}")
                raise
    
    def _initialize_embeddings(self):
        """Initialize embeddings based on settings and mode"""
        try:
            # Use Azure OpenAI embeddings if in production mode and configured
            if self.mode == "prod" and settings.is_azure_openai_configured():
                # Production mode with Azure OpenAI embeddings
                try:
                    from langchain_openai import AzureOpenAIEmbeddings
                    self.embedding_model = AzureOpenAIEmbeddings(
                        azure_deployment=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
                        api_version=settings.AZURE_OPENAI_API_VERSION,
                        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                        api_key=settings.AZURE_OPENAI_API_KEY
                    )
                    print(f"📊 Production: Using Azure OpenAI embeddings ({settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT})")
                except ImportError:
                    print("⚠️ langchain_openai not available, falling back to HuggingFace")
                    self._init_huggingface_embeddings()
            elif settings.is_openai_provider() and settings.is_openai_configured() and self.mode == "prod":
                # Fallback to standard OpenAI embeddings if Azure not configured
                try:
                    from langchain_openai import OpenAIEmbeddings
                    self.embedding_model = OpenAIEmbeddings(
                        model=settings.OPENAI_EMBEDDING_MODEL,
                        api_key=settings.OPENAI_API_KEY
                    )
                    print(f"📊 Production: Using OpenAI embeddings ({settings.OPENAI_EMBEDDING_MODEL})")
                except ImportError:
                    print("⚠️ langchain_openai not available, falling back to HuggingFace")
                    self._init_huggingface_embeddings()
            else:
                # Development mode or HuggingFace preference
                self._init_huggingface_embeddings()
                
        except Exception as e:
            print(f"⚠️ Embedding initialization failed: {e}, using fallback")
            self._init_huggingface_embeddings()
    
    def _init_huggingface_embeddings(self):
        """Initialize HuggingFace embeddings as fallback"""
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'}
            )
            print(f"📊 Using HuggingFace embeddings ({settings.EMBEDDING_MODEL})")
        except ImportError:
            print("⚠️ HuggingFaceEmbeddings not available, trying sentence-transformers")
            try:
                from sentence_transformers import SentenceTransformer
                self.st_model = SentenceTransformer(settings.EMBEDDING_MODEL)
                self.embedding_model = None
                print(f"📊 Using SentenceTransformer directly ({settings.EMBEDDING_MODEL})")
            except ImportError:
                raise ImportError("Please install: pip install sentence-transformers")
    
    def _embed_query(self, text: str) -> List[float]:
        """Create embedding for text query"""
        try:
            if self.embedding_model:
                return self.embedding_model.embed_query(text)
            else:
                # Fallback to direct SentenceTransformer
                return self.st_model.encode([text])[0].tolist()
        except Exception as e:
            print(f"⚠️ Embedding failed: {e}")
            return []
    
    def _setup_tools(self):
        """Setup the three specialized tools for ReAct agent"""
        
        # Tool 1: Internal Documents Search
        def search_internal_docs(query: str) -> str:
            """Search internal company documents"""
            try:
                if not self.client:
                    return "❌ Internal docs search unavailable - no Qdrant client"
                
                # Create query embedding
                query_vector = self._embed_query(query)
                if not query_vector:
                    return "❌ Could not create embedding for internal docs search"
                
                # Search in DATA collection
                results = self.client.search(
                    collection_name=self.DATA_COLLECTION,
                    query_vector=query_vector,
                    limit=3,
                    with_payload=True
                )
                
                if not results:
                    return "📄 No relevant internal documents found"
                
                # Format results
                docs = []
                for result in results:
                    content = result.payload.get('content', '')[:200]
                    filename = result.payload.get('filename', 'Unknown')
                    score = f"({result.score:.2f})"
                    docs.append(f"📄 {filename} {score}: {content}...")
                
                return f"✅ Found {len(docs)} internal documents:\n" + "\n".join(docs)
            
            except Exception as e:
                return f"❌ Error searching internal docs: {e}"
        
        # Tool 2: RFP History Search
        def search_rfp_history(query: str) -> str:
            """Search past RFP question-answer pairs"""
            try:
                if not self.client:
                    return "❌ RFP history search unavailable - no Qdrant client"
                
                # Create query embedding
                query_vector = self._embed_query(query)
                if not query_vector:
                    return "❌ Could not create embedding for RFP history search"
                
                # Search in RFP collection
                results = self.client.search(
                    collection_name=self.RFP_COLLECTION,
                    query_vector=query_vector,
                    limit=3,
                    with_payload=True
                )
                
                if not results:
                    return "🔍 No similar RFP questions found in history"
                
                # Format results
                rfps = []
                for result in results:
                    question = result.payload.get('question', '')[:100]
                    answer = result.payload.get('answer', '')[:150]
                    score = f"({result.score:.2f})"
                    rfps.append(f"🔍 RFP {score}:\nQ: {question}...\nA: {answer}...")
                
                return f"✅ Found {len(rfps)} similar RFP pairs:\n" + "\n".join(rfps)
            
            except Exception as e:
                return f"❌ Error searching RFP history: {e}"
        
        # Tool 3: Web Search
        def search_web(query: str) -> str:
            """Search the web for current information"""
            try:
                search = DuckDuckGoSearchRun()
                enhanced_query = f"{query} SaaS technology enterprise"
                results = search.run(enhanced_query)
                
                if not results or len(results) < 20:
                    return "🌐 No relevant web information found"
                
                # Limit length to prevent token overflow
                if len(results) > 400:
                    results = results[:400] + "..."
                
                return f"✅ Web search results: {results}"
            
            except Exception as e:
                return f"❌ Error in web search: {e}"
        
        # Define tools for the agent
        self.tools = [
            Tool(
                name="search_internal_docs",
                func=search_internal_docs,
                description="Search internal company documents, policies, and procedures. Use for company-specific questions about security, processes, or technical capabilities."
            ),
            Tool(
                name="search_rfp_history",
                func=search_rfp_history,
                description="Search past RFP questions and answers to maintain consistency. Use for questions that might have been answered before."
            ),
            Tool(
                name="search_web",
                func=search_web,
                description="Search the web for current SaaS industry information, standards, or technical details. Use for technical or market questions needing current info."
            )
        ]
    
    def _setup_agent(self):
        """Setup the ReAct agent with Ollama"""
        try:
            self.agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                max_iterations=3,  # Limit for efficiency
                early_stopping_method="generate",
                handle_parsing_errors=True
            )
            print("🎯 ReAct agent configured successfully")
        except Exception as e:
            print(f"❌ Failed to setup ReAct agent: {e}")
            raise
    
    def _parse_structured_response(self, response: str) -> Dict[str, str]:
        """
        Parse structured response to extract Answer (Yes/No) and Comments
        Even handles responses with errors or incomplete parsing
        
        Args:
            response: Raw response from the LLM
            
        Returns:
            Dict with 'answer' and 'comments' keys
        """
        try:
            # Initialize default values
            answer = ""
            comments = ""
            
            # Handle error messages that contain the actual response
            if "Could not parse LLM output:" in response:
                # Extract the content between backticks
                import re
                pattern = r"`([^`]*ANSWER:[^`]*COMMENTS:[^`]*)`"
                match = re.search(pattern, response, re.DOTALL)
                if match:
                    response = match.group(1)
                    print(f"🔧 Extracted response from error message")
            
            # Look for ANSWER: and COMMENTS: patterns
            lines = response.split('\n')
            current_section = None
            comments_lines = []
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('ANSWER:'):
                    answer = line.replace('ANSWER:', '').strip()
                    current_section = 'answer'
                elif line.startswith('COMMENTS:'):
                    comments = line.replace('COMMENTS:', '').strip()
                    current_section = 'comments'
                    if comments:  # If there's content on the same line
                        comments_lines = []  # Reset to avoid duplicates
                elif current_section == 'comments' and line and not line.startswith('For troubleshooting'):
                    # Continue collecting comments if we're in the comments section
                    # Skip troubleshooting messages
                    comments_lines.append(line)
            
            # Join additional comment lines
            if comments_lines:
                full_comments = comments + ' ' + ' '.join(comments_lines) if comments else ' '.join(comments_lines)
                comments = full_comments.strip()
            
            # Alternative parsing if direct method didn't work
            if not answer or not comments:
                # Try regex patterns
                import re
                answer_match = re.search(r'ANSWER:\s*(Yes|No)', response, re.IGNORECASE)
                if answer_match:
                    answer = answer_match.group(1).capitalize()
                
                comments_match = re.search(r'COMMENTS:\s*(.+?)(?=For troubleshooting|$)', response, re.DOTALL | re.IGNORECASE)
                if comments_match:
                    comments = comments_match.group(1).strip()
            
            # Validate and clean answer
            if answer:
                if answer.lower() in ['yes', 'y', 'oui', 'true', '1']:
                    answer = 'Yes'
                elif answer.lower() in ['no', 'n', 'non', 'false', '0']:
                    answer = 'No'
                elif 'yes' in answer.lower():
                    answer = 'Yes'
                elif 'no' in answer.lower():
                    answer = 'No'
            
            # If still no answer, try to determine from comments content
            if not answer and comments:
                if any(word in comments.lower() for word in ['can', 'able', 'support', 'comply', 'provide', 'implement', 'yes', 'available']):
                    answer = 'Yes'
                else:
                    answer = 'No'
            
            # Final fallback
            if not answer:
                answer = 'No'
            
            # Ensure comments exist and are clean
            if not comments:
                comments = "No detailed explanation was provided by the AI system."
            else:
                # Clean up troubleshooting messages
                comments = re.sub(r'For troubleshooting.*$', '', comments, flags=re.DOTALL).strip()
            
            print(f"✅ Parsed - Answer: {answer}, Comments length: {len(comments)} chars")
            return {
                'answer': answer,
                'comments': comments
            }
            
        except Exception as e:
            print(f"⚠️ Failed to parse structured response: {e}")
            # Try one more time with simple regex
            try:
                import re
                answer_match = re.search(r'ANSWER:\s*(Yes|No)', response, re.IGNORECASE)
                comments_match = re.search(r'COMMENTS:\s*(.+)', response, re.DOTALL | re.IGNORECASE)
                
                answer = answer_match.group(1).capitalize() if answer_match else 'No'
                comments = comments_match.group(1).strip() if comments_match else f"Parsing error: {str(e)}"
                
                # Clean comments
                comments = re.sub(r'For troubleshooting.*$', '', comments, flags=re.DOTALL).strip()
                
                return {
                    'answer': answer,
                    'comments': comments if comments else "Could not extract detailed explanation."
                }
            except:
                return {
                    'answer': 'No',
                    'comments': f"Error parsing response. Raw content: {response[:200]}..."
                }
    
    def answer_rfp_question(self, question: str) -> Dict[str, Any]:
        """
        Answer an RFP question using ReAct reasoning with Yes/No format
        
        Args:
            question: The RFP question to answer
            
        Returns:
            Dict containing structured answer and metadata
        """
        try:
            # Create a focused prompt for RFP context with structured output
            prompt = f"""
You are answering an RFP (Request for Proposal) question for a SaaS company.

Question: {question}

Instructions:
1. Think about what information you need to answer this question properly
2. Use the available tools to gather relevant information:
   - search_internal_docs: For company policies and internal information
   - search_rfp_history: For consistency with previous RFP responses
   - search_web: For current industry standards or technical information
3. Based on your research, determine if our company can satisfy this requirement/question

CRITICAL: You MUST end your response with this EXACT format:
ANSWER: Yes
COMMENTS: [Your detailed explanation here]

OR

ANSWER: No  
COMMENTS: [Your detailed explanation here]

Do not include any other text after the ANSWER and COMMENTS sections.

Example:
ANSWER: Yes
COMMENTS: Our platform implements enterprise-grade encryption (AES-256) for data at rest and TLS 1.3 for data in transit. We maintain SOC 2 Type II compliance and undergo annual security audits. Our security framework includes multi-factor authentication, role-based access controls, and continuous monitoring systems.
"""
            
            print(f"🤖 Processing: {question[:50]}...")
            
            # Run the ReAct agent
            response = self.agent.run(prompt)
            
            # Parse structured response
            parsed_response = self._parse_structured_response(response)
            
            # Save successful Q&A pairs
            if parsed_response['answer'] and len(parsed_response['comments']) > 20:
                metadata = {
                    'method': f'react_{self.mode}',
                    'model': self.model,
                    'mode': self.mode,
                    'structured_answer': parsed_response['answer'],
                    'structured_comments': parsed_response['comments']
                }
                self.save_qa_pair(question, f"ANSWER: {parsed_response['answer']}\nCOMMENTS: {parsed_response['comments']}", metadata)
            
            return {
                'question': question,
                'answer': parsed_response['answer'],  # Yes/No
                'comments': parsed_response['comments'],  # Detailed explanation
                'full_response': response,  # Original full response
                'method': f'react_{self.mode}',
                'model': self.model,
                'mode': self.mode,
                'timestamp': datetime.now().isoformat(),
                'sources_used': 'react_reasoning'
            }
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error processing question: {error_msg}")
            
            # Check if the error contains the actual LLM response with ANSWER/COMMENTS
            if "Could not parse LLM output:" in error_msg and "`" in error_msg:
                print(f"🔧 Attempting to extract answer from error message...")
                # Try to parse from the error message
                parsed_response = self._parse_structured_response(error_msg)
                
                if parsed_response['answer'] and parsed_response['comments'] and len(parsed_response['comments']) > 20:
                    print(f"✅ Successfully extracted answer from error message")
                    return {
                        'question': question,
                        'answer': parsed_response['answer'],  # Yes/No
                        'comments': parsed_response['comments'],  # Detailed explanation
                        'full_response': error_msg,  # Original error with response
                        'method': f'react_{self.mode}_recovered',
                        'model': self.model,
                        'mode': self.mode,
                        'timestamp': datetime.now().isoformat(),
                        'sources_used': 'react_reasoning_recovered'
                    }
            
            # Fallback if no valid response could be extracted
            print(f"⚠️ Manual review required.")
            return {
                'question': question,
                'answer': 'No',  # Default to No for errors
                'comments': f"Error processing this question: {str(e)[:150]}. Manual review required.",
                'full_response': error_msg,
                'method': f'react_{self.mode}_error',
                'model': self.model,
                'mode': self.mode,
                'timestamp': datetime.now().isoformat()
            }
    
    def batch_answer_questions(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Answer multiple RFP questions using ReAct
        
        Args:
            questions: List of RFP questions
            
        Returns:
            List of answer dictionaries
        """
        results = []
        
        print(f"🎯 Starting ReAct batch processing for {len(questions)} questions")
        
        for i, question in enumerate(questions, 1):
            print(f"\n📝 Question {i}/{len(questions)}")
            
            if not question or len(question.strip()) < 10:
                print("⏭️  Skipping short/empty question")
                continue
            
            result = self.answer_rfp_question(question)
            results.append(result)
            
            print(f"✅ Question {i} processed")
        
        print(f"\n🎉 ReAct batch processing complete: {len(results)} answers generated")
        return results
    
    def save_qa_pair(self, question: str, answer: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Save Q&A pair to RFP collection
        
        Args:
            question: The RFP question
            answer: Generated answer
            metadata: Additional metadata
            
        Returns:
            True if saved successfully
        """
        if not self.client:
            print("⚠️ Cannot save Q&A pair - missing client")
            return False
            
        try:
            # Create embedding for the question
            question_vector = self._embed_query(question)
            if not question_vector:
                print("⚠️ Could not create embedding, skipping save")
                return False
            
            # Prepare payload
            payload = {
                'question': question,
                'answer': answer,
                'method': f'react_{self.mode}',
                'model': self.model,
                'mode': self.mode,
                'timestamp': datetime.now().isoformat(),
                'validated': False,  # Mark as AI-generated
                **(metadata or {})
            }
            
            # Generate unique point ID
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"react_{self.mode}_{question}_{datetime.now().isoformat()}"))
            
            # Store in RFP collection
            self.client.upsert(
                collection_name=self.RFP_COLLECTION,
                points=[{
                    "id": point_id,
                    "vector": question_vector,
                    "payload": payload
                }]
            )
            
            print(f"💾 Saved ReAct Q&A pair to database (mode: {self.mode})")
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to save Q&A pair: {e}")
            return False


# Example usage and testing
def test_react_retriever(mode="dev"):
    """Test the ReAct retriever with sample questions"""
    print(f"🧪 Testing ReAct RFP Retriever in {mode.upper()} mode")
    
    # Initialize retriever
    try:
        retriever = ReactRFPRetriever(mode=mode)
        print(f"✅ ReAct retriever initialized successfully in {mode} mode")
    except Exception as e:
        print(f"❌ Failed to initialize retriever: {e}")
        return None
    
    # Test questions
    test_questions = [
        "What is your company's data security policy?",
        "How do you handle GDPR compliance?",
        "What SaaS architecture do you use for scalability?"
    ]
    
    print(f"\n🔍 Testing with {len(test_questions)} questions:")
    
    # Test single question first
    print("\n" + "="*60)
    print(f"Testing single question in {mode} mode:")
    result = retriever.answer_rfp_question(test_questions[0])
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")  # Yes/No
    print(f"Comments: {result['comments'][:200]}...")  # Explanation
    print(f"Method: {result['method']}")
    print(f"Model: {result['model']}")
    print(f"Mode: {result['mode']}")
    print(f"Full Response Length: {len(result.get('full_response', ''))} chars")
    
    # Test batch processing (limit for testing)
    print("\n" + "="*60)
    print(f"Testing batch processing in {mode} mode:")
    results = retriever.batch_answer_questions(test_questions[:2])  # Limit for testing
    
    print(f"\n🎉 Test complete! Processed {len(results)} questions in {mode} mode")
    
    return retriever

def test_both_modes():
    """Test both development and production modes"""
    print("🧪 Testing ReAct RFP Retriever - Both Modes")
    print("="*60)
    
    # Test development mode
    print("\n🛠️ DEVELOPMENT MODE TEST")
    print("-" * 40)
    dev_retriever = test_react_retriever("dev")
    
    # Check if OpenAI is available for production test
    if settings.OPENAI_API_KEY:
        print("\n🚀 PRODUCTION MODE TEST")
        print("-" * 40)
        prod_retriever = test_react_retriever("prod")
    else:
        print("\n⚠️ PRODUCTION MODE SKIPPED")
        print("Set OPENAI_API_KEY in .env to test production mode")
    
    print("\n🎉 Mode comparison test complete!")

if __name__ == "__main__":
    import sys
    
    # Check command line arguments for mode selection
    if len(sys.argv) > 1 and sys.argv[1] in ["dev", "prod", "both"]:
        mode = sys.argv[1]
        if mode == "both":
            test_both_modes()
        else:
            test_react_retriever(mode)
    else:
        print("Usage: python react_retriever.py [dev|prod|both]")
        print("Testing in development mode by default...")
        test_react_retriever("dev")
