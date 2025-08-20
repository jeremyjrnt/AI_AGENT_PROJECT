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
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
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
        """Initialize LLM - Azure OpenAI only for production mode"""
        if self.mode == "prod":
            # Production mode: Use Azure OpenAI
            try:
                from langchain_openai import AzureChatOpenAI
                
                if not settings.AZURE_OPENAI_API_KEY:
                    raise ValueError("Azure OpenAI API key not found. Set AZURE_OPENAI_API_KEY in .env file.")
                
                self.llm = AzureChatOpenAI(
                    api_key=settings.AZURE_OPENAI_API_KEY,
                    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                    deployment_name=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
                    api_version=settings.AZURE_OPENAI_API_VERSION,
                    temperature=0.1
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
                
                # Format results using direct answer and comments from metadata
                rfps = []
                for result in results:
                    question = result.payload.get('question', '')[:100]
                    score = f"({result.score:.2f})"
                    
                    # Extract answer and comments directly from metadata
                    answer = result.payload.get('answer', '')
                    comments = result.payload.get('comments', '')
                    
                    # If direct fields not available, try structured fields
                    if not answer:
                        answer = result.payload.get('structured_answer', 'No answer available')
                    if not comments:
                        comments = result.payload.get('structured_comments', 'No comments available')
                    
                    # Limit lengths for display
                    answer_display = answer[:100] if answer else 'No answer'
                    comments_display = comments[:200] if comments else 'No comments'
                    
                    # Format the result with answer and comments from the k nearest neighbors
                    rfps.append(f"🔍 RFP {score}:\nQ: {question}...\nAnswer: {answer_display}...\nComments: {comments_display}...")
                
                return f"✅ Found {len(rfps)} similar RFP pairs from k-NN search:\n" + "\n".join(rfps)
            
            except Exception as e:
                return f"❌ Error searching RFP history: {e}"
        
        # Tool 3: Web Search with DuckDuckGo
        def search_web(query: str) -> str:
            """Search the web for current information using DuckDuckGo"""
            try:
                search = DuckDuckGoSearchRun()
                enhanced_query = f"{query} SaaS technology enterprise standards best practices"
                results = search.run(enhanced_query)
                
                if not results or len(results) < 20:
                    return "🌐 No relevant web information found"
                
                # Limit length to prevent token overflow
                if len(results) > 400:
                    results = results[:400] + "..."
                
                return f"✅ Web search results: {results}"
            
            except Exception as e:
                # Simply handle the exception and return empty result
                print(f"⚠️ Web search failed: {e}")
                return "🌐 Web search temporarily unavailable"
        
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
        """Setup the ReAct agent with classic LangChain structure"""
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
            print("🎯 ReAct agent configured successfully with classic structure")
        except Exception as e:
            print(f"❌ Failed to setup ReAct agent: {e}")
            raise
    
    def answer_rfp_question(self, question: str) -> Dict[str, Any]:
        """
        Process an RFP question through ReAct agent then refine with Azure OpenAI
        
        Args:
            question: The RFP question to process
            
        Returns:
            dict: Structured response with answer, comments, method, mode, and question_vector
        """
        print(f"🤖 Processing: {question[:50]}...")
        
        # Embed the question and keep the vector
        question_vector = self._embed_query(question)
        
        try:
            # Step 1: Get raw response from ReAct agent
            print("🔄 Step 1: Running ReAct agent...")
            agent_response = self.agent.run(question)
            print(f"✅ ReAct agent completed")
            
            # Step 2: Post-process with Azure OpenAI for structured output
            print("🔄 Step 2: Structuring response with Azure OpenAI...")
            structured_response = self._structure_response_with_llm(question, agent_response)
            
            return {
                "answer": structured_response.get("answer", "Unknown"),
                "comments": structured_response.get("comments", ""),
                "method": "react_with_llm_structuring",
                "mode": self.mode,
                "raw_agent_response": agent_response[:500] + "..." if len(agent_response) > 500 else agent_response,
                "question_vector": question_vector,
                "question": question
            }
            
        except Exception as e:
            print(f"❌ Error processing question: {e}")
            # Standard error response as requested
            return {
                "answer": "No",
                "comments": "Please review this section",
                "method": "error_fallback",
                "mode": self.mode,
                "raw_agent_response": "",
                "question_vector": question_vector,
                "question": question
            }
    
    def _structure_response_with_llm(self, original_question: str, agent_response: str) -> Dict[str, str]:
        """
        Use Azure OpenAI to structure the ReAct agent response into Answer/Comments format
        
        Args:
            original_question: The original RFP question
            agent_response: Raw response from ReAct agent
            
        Returns:
            dict: Structured response with 'answer' and 'comments' keys
        """
        
        # Create structured prompt for Azure OpenAI
        structuring_prompt = f"""You are an expert Okta employee at analyzing RFP (Request for Proposal) responses and providing structured answers.

ORIGINAL QUESTION:
{original_question}

RAW RESEARCH RESPONSE:
{agent_response}

Your task is to analyze the research response above and provide a structured answer for the RFP question.

REQUIREMENTS:
1. Answer: Must be either "Yes" or "No"
2. Comments: Provide concise but detailed justification, context, and supporting information

FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS:
Answer: [Yes/No]
Comments: [Detailed but concise explanation supporting your answer, including relevant technical details, standards supported, implementation considerations, and any limitations or requirements]

Focus on being accurate, professional, and providing sufficient detail for RFP evaluators to understand the capability and implementation. If the response is incomplete or unclear, just say 'Please review this section.' in 'Comments'."""

        try:
            if self.mode == "prod":
                # Use Azure OpenAI for structuring
                from langchain.schema import HumanMessage
                
                response = self.llm.invoke([HumanMessage(content=structuring_prompt)])
                structured_text = response.content
                
                # Parse the structured response
                return self._parse_structured_response(structured_text)
                
            else:
                # Dev mode fallback - simple parsing of agent response
                return {
                    "answer": "Yes" if any(word in agent_response.lower() for word in ["yes", "support", "provide", "enable"]) else "No",
                    "comments": agent_response[:500] + "..." if len(agent_response) > 500 else agent_response
                }
                
        except Exception as e:
            print(f"⚠️ Error in LLM structuring: {e}")
            # Standard error response as requested
            return {
                "answer": "No",
                "comments": "Please review this section"
            }
    
    def _parse_structured_response(self, structured_text: str) -> Dict[str, str]:
        """
        Parse the structured response from Azure OpenAI
        
        Args:
            structured_text: The formatted response from Azure OpenAI
            
        Returns:
            dict: Parsed response with 'answer' and 'comments' keys
        """
        
        lines = structured_text.strip().split('\n')
        answer = "Unknown"
        comments = ""
        
        collecting_comments = False
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("Answer:"):
                answer = line.replace("Answer:", "").strip()
                collecting_comments = False
                
            elif line.startswith("Comments:"):
                comments = line.replace("Comments:", "").strip()
                collecting_comments = True
                
            elif collecting_comments and line:
                # Continue collecting comments on subsequent lines
                comments += " " + line
        
        # Clean up the response and handle malformed responses
        if answer not in ["Yes", "No", "Partial"]:
            # If response is not properly structured, use standard error response
            answer = "No"
            comments = "Please review this section"
        
        if not comments or comments.strip() == "":
            comments = "Please review this section"
        
        return {
            "answer": answer,
            "comments": comments
        }
    
    def process_rfp_excel(self, excel_path: str, output_path: str = None) -> tuple[str, list]:
        """
        Process an Excel file with RFP questions and fill Answer/Comments columns
        
        Args:
            excel_path: Path to the Excel file with columns 'Question', 'Answer', 'Comments'
            output_path: Optional path for output file (defaults to adding '_processed' to input)
            
        Returns:
            tuple: (Path to the processed Excel file, List of question vectors and metadata)
        """
        try:
            import pandas as pd
            from pathlib import Path
            
            print(f"📊 Processing Excel file: {excel_path}")
            
            # Load the Excel file
            df = pd.read_excel(excel_path)
            
            # Validate required columns
            required_columns = ['Question', 'Answer', 'Comments']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            print(f"✅ Excel loaded with {len(df)} questions")
            
            # Set output path if not provided
            if output_path is None:
                input_path = Path(excel_path)
                output_path = input_path.parent / f"{input_path.stem}_processed{input_path.suffix}"
            
            # Process each question and collect vectors
            processed_count = 0
            errors = []
            question_vectors = []  # Store vectors and metadata for vector DB
            
            for index, row in df.iterrows():
                question = row['Question']
                
                # Skip if question is empty or Answer already filled
                if pd.isna(question) or not question.strip():
                    print(f"⏭️ Skipping row {index + 1}: Empty question")
                    continue
                
                if pd.notna(row['Answer']) and row['Answer'].strip():
                    print(f"⏭️ Skipping row {index + 1}: Answer already exists")
                    continue
                
                try:
                    print(f"\n🤖 Processing question {index + 1}/{len(df)}: {question[:60]}...")
                    
                    # Use the ReAct agent to process the question
                    result = self.answer_rfp_question(question)
                    
                    # Update the dataframe
                    df.at[index, 'Answer'] = result['answer']
                    df.at[index, 'Comments'] = result['comments']
                    
                    # Collect vector and metadata for vector DB
                    if result.get('question_vector'):
                        question_vectors.append({
                            'vector': result['question_vector'],
                            'question': result['question'],
                            'answer': result['answer'],
                            'comments': result['comments'],
                            'row_index': index
                        })
                    
                    processed_count += 1
                    print(f"✅ Question {index + 1} completed: {result['answer']}")
                    
                    # Save intermediate progress every 5 questions
                    if processed_count % 5 == 0:
                        df.to_excel(output_path, index=False)
                        print(f"💾 Intermediate save: {processed_count} questions processed")
                    
                except Exception as e:
                    error_msg = f"Error processing question {index + 1}: {str(e)}"
                    errors.append(error_msg)
                    print(f"❌ {error_msg}")
                    
                    # Use standard error response as requested
                    df.at[index, 'Answer'] = "No"
                    df.at[index, 'Comments'] = "Please review this section"
                    
                    # Still try to collect vector for failed questions (if embedding worked)
                    try:
                        question_vector = self._embed_query(question)
                        if question_vector:
                            question_vectors.append({
                                'vector': question_vector,
                                'question': question,
                                'answer': "No",
                                'comments': "Please review this section",
                                'row_index': index
                            })
                    except:
                        pass  # Skip vector collection if embedding also failed
            
            # Final save
            df.to_excel(output_path, index=False)
            
            # Summary report
            print(f"\n📊 PROCESSING COMPLETE!")
            print(f"✅ Successfully processed: {processed_count} questions")
            print(f"❌ Errors encountered: {len(errors)}")
            print(f"🔢 Question vectors collected: {len(question_vectors)}")
            print(f"💾 Output saved to: {output_path}")
            
            if errors:
                print(f"\n🔍 Error details:")
                for error in errors[:5]:  # Show first 5 errors
                    print(f"   - {error}")
                if len(errors) > 5:
                    print(f"   ... and {len(errors) - 5} more errors")
            
            return str(output_path), question_vectors
            
        except ImportError:
            raise ImportError("pandas required for Excel processing. Run: pip install pandas openpyxl")
        except Exception as e:
            print(f"❌ Error processing Excel file: {e}")
            raise
    
    def process_rfp_excel_batch(self, excel_path: str, batch_size: int = 3, output_path: str = None) -> tuple[str, list]:
        """
        Process an Excel file in batches to avoid overwhelming the API
        
        Args:
            excel_path: Path to the Excel file
            batch_size: Number of questions to process in each batch
            output_path: Optional path for output file
            
        Returns:
            tuple: (Path to the processed Excel file, List of question vectors and metadata)
        """
        try:
            import pandas as pd
            import time
            from pathlib import Path
            
            print(f"📊 Processing Excel file in batches of {batch_size}: {excel_path}")
            
            # Load the Excel file
            df = pd.read_excel(excel_path)
            
            # Validate required columns
            required_columns = ['Question', 'Answer', 'Comments']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            print(f"✅ Excel loaded with {len(df)} questions")
            
            # Set output path if not provided
            if output_path is None:
                input_path = Path(excel_path)
                output_path = input_path.parent / f"{input_path.stem}_processed{input_path.suffix}"
            
            # Get questions that need processing
            questions_to_process = []
            for index, row in df.iterrows():
                question = row['Question']
                if (not pd.isna(question) and question.strip() and 
                    (pd.isna(row['Answer']) or not row['Answer'].strip())):
                    questions_to_process.append(index)
            
            print(f"📋 {len(questions_to_process)} questions need processing")
            
            # Process in batches
            processed_count = 0
            batch_count = 0
            question_vectors = []  # Store vectors and metadata for vector DB
            
            for i in range(0, len(questions_to_process), batch_size):
                batch_indices = questions_to_process[i:i + batch_size]
                batch_count += 1
                
                print(f"\n🔄 Processing batch {batch_count} ({len(batch_indices)} questions)...")
                
                for index in batch_indices:
                    question = df.at[index, 'Question']
                    
                    try:
                        print(f"🤖 Question {index + 1}: {question[:50]}...")
                        
                        # Use the ReAct agent
                        result = self.answer_rfp_question(question)
                        
                        # Update the dataframe
                        df.at[index, 'Answer'] = result['answer']
                        df.at[index, 'Comments'] = result['comments']
                        
                        # Collect vector and metadata for vector DB
                        if result.get('question_vector'):
                            question_vectors.append({
                                'vector': result['question_vector'],
                                'question': result['question'],
                                'answer': result['answer'],
                                'comments': result['comments'],
                                'row_index': index
                            })
                        
                        processed_count += 1
                        print(f"✅ Completed: {result['answer']}")
                        
                    except Exception as e:
                        print(f"❌ Error: {e}")
                        # Use standard error response as requested
                        df.at[index, 'Answer'] = "No"
                        df.at[index, 'Comments'] = "Please review this section"
                        
                        # Still try to collect vector for failed questions
                        try:
                            question_vector = self._embed_query(question)
                            if question_vector:
                                question_vectors.append({
                                    'vector': question_vector,
                                    'question': question,
                                    'answer': "No",
                                    'comments': "Please review this section",
                                    'row_index': index
                                })
                        except:
                            pass  # Skip vector collection if embedding also failed
                
                # Save after each batch
                df.to_excel(output_path, index=False)
                print(f"💾 Batch {batch_count} saved. Total processed: {processed_count}")
                
                # Small delay between batches to avoid overwhelming the API
                if i + batch_size < len(questions_to_process):
                    print("⏸️ Brief pause between batches...")
                    time.sleep(2)
            
            print(f"\n🎉 BATCH PROCESSING COMPLETE!")
            print(f"📊 Processed {processed_count} questions in {batch_count} batches")
            print(f"🔢 Question vectors collected: {len(question_vectors)}")
            print(f"💾 Final output: {output_path}")
            
            return str(output_path), question_vectors
            
        except ImportError:
            raise ImportError("pandas required for Excel processing. Run: pip install pandas openpyxl")
        except Exception as e:
            print(f"❌ Error in batch processing: {e}")
            raise

