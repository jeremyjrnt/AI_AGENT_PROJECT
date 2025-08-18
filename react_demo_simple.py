#!/usr/bin/env python3
"""
ReAct RFP Retriever - Version Simplifiée pour Démo
Sans embeddings HuggingFace pour éviter les téléchargements
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# LangChain imports for ReAct
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun

def create_simple_react_demo():
    """Créer une démo ReAct simplifiée"""
    
    print("🤖 Creating Simple ReAct Demo with Ollama")
    
    # Initialize Ollama LLM
    try:
        llm = ChatOllama(
            model="qwen2:0.5b",
            temperature=0.1,
            base_url="http://localhost:11434"
        )
        print("✅ Ollama LLM initialized")
    except Exception as e:
        print(f"❌ Ollama initialization failed: {e}")
        return None
    
    # Create simple mock tools for demo
    def mock_internal_search(query: str) -> str:
        """Mock internal documents search"""
        return f"📄 Internal docs found for '{query}': Company security policy includes encryption standards, access controls, and audit procedures. Document source: internal_security_v2.pdf"
    
    def mock_rfp_search(query: str) -> str:
        """Mock RFP history search"""  
        return f"🔍 RFP history for '{query}': Previously answered similar question in RFP-2024-001 with response about compliance frameworks and security certifications."
    
    def web_search(query: str) -> str:
        """Real web search"""
        try:
            search = DuckDuckGoSearchRun()
            results = search.run(f"{query} SaaS enterprise")
            return f"🌐 Web results: {results[:200]}..."
        except Exception as e:
            return f"🌐 Web search failed: {e}"
    
    # Define tools
    tools = [
        Tool(
            name="search_internal_docs",
            func=mock_internal_search,
            description="Search internal company documents for policies and procedures"
        ),
        Tool(
            name="search_rfp_history", 
            func=mock_rfp_search,
            description="Search past RFP responses to maintain consistency"
        ),
        Tool(
            name="search_web",
            func=web_search,
            description="Search the web for current industry information"
        )
    ]
    
    # Create ReAct agent
    try:
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=4,
            handle_parsing_errors=True
        )
        print("✅ ReAct agent created successfully")
        return agent
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return None

def test_react_demo():
    """Test de la démo ReAct"""
    
    print("\n🧪 Testing ReAct Demo")
    
    agent = create_simple_react_demo()
    if not agent:
        print("❌ Cannot create agent")
        return
    
    # Test with a question that should trigger tool usage
    question = "What is our company's data security approach and how does it compare to industry standards?"
    
    # Create a very directive prompt
    prompt = f"""
You are answering an RFP question for a SaaS company.

Question: {question}

To answer this properly, you MUST:
1. Use search_internal_docs to find our company's security policies
2. Use search_rfp_history to see how we've answered similar questions before
3. Use search_web to compare with industry standards
4. Combine all findings into a comprehensive answer

Do NOT skip any tools. You must use all three tools to gather complete information.
"""
    
    print(f"Question: {question}")
    print("\nRunning ReAct agent...")
    print("=" * 70)
    
    try:
        response = agent.run(prompt)
        print("=" * 70)
        print(f"Final Answer: {response}")
        return response
    except Exception as e:
        print(f"❌ Agent execution failed: {e}")
        return None

if __name__ == "__main__":
    print("🚀 ReAct Demo - Free Version with Ollama")
    response = test_react_demo()
    
    if response:
        print("\n✅ ReAct Demo completed successfully!")
        print(f"Response preview: {response[:200]}...")
    else:
        print("\n❌ Demo failed")
