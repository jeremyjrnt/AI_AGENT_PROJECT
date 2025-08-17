import os
import sys
from parsers.rfp_parser import read_pdf_text, extract_and_export

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py path/to/file.pdf [model_name]")
        print("Models: 'pattern' (default, free) or 'gpt-3.5-turbo' (requires OpenAI API key)")
        sys.exit(1)

    pdf_path = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "pattern"
    
    if not os.path.isfile(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    try:
        # Get OpenAI API key from environment if using GPT model
        openai_api_key = None
        if model_name.startswith("gpt"):
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                print("⚠️ OpenAI API key not found in environment variables.")
                print("Please set OPENAI_API_KEY in your .env file or use 'pattern' model.")
                print("Falling back to pattern-based extraction...")
                model_name = "pattern"
        
        print(f"🔧 Using model: {model_name}")
        
        # Extract text from PDF
        text = read_pdf_text(pdf_path)
        
        # Extract questions and export to Excel
        questions, out_path = extract_and_export(text, model_name=model_name, openai_api_key=openai_api_key)
        
        print(f"\n📊 RESULTS:")
        print(f"✅ Questions extracted: {len(questions)}")
        print(f"✅ Excel written to: {out_path}")
        
        # Show sample questions
        if questions:
            print(f"\n📄 Sample questions:")
            for i, q in enumerate(questions[:3], 1):
                print(f"  {i}. [{q.get('question_type', 'N/A')}] {q.get('question_text', '')[:80]}...")
                
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
