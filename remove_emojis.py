#!/usr/bin/env python3
"""
Script to remove all emojis from the RFP Manager UI and make it professional
"""

import re
from pathlib import Path

def remove_emojis_from_file(file_path):
    """Remove emojis from a file and make the interface professional"""
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Define specific emoji patterns to replace - be very careful with context
    replacements = [
        # Tab names - exact matches
        ('st.tabs(["📋 RFP Manager", "📚 How to Use"])', 'st.tabs(["RFP Manager", "User Guide"])'),
        
        # Headers - exact patterns
        ('st.title("🚀 RFP Question Manager")', 'st.title("RFP Question Manager")'),
        ('st.header("📋 RFP Processing Workflow")', 'st.header("RFP Processing Workflow")'),
        ('st.sidebar.header("📁 RFP File Management")', 'st.sidebar.header("File Management")'),
        ('st.sidebar.header("📊 RFP Statistics")', 'st.sidebar.header("RFP Statistics")'),
        ('st.sidebar.header("🤖 AI Processing Mode")', 'st.sidebar.header("AI Processing Mode")'),
        
        # Buttons
        ('st.sidebar.button("🔄 Index Files in Vector DB"', 'st.sidebar.button("Index Files in Vector DB"'),
        ('st.sidebar.button("🧹 Force Cleanup"', 'st.sidebar.button("Force Cleanup"'),
        ('st.sidebar.button("🔍 Inspect Collection"', 'st.sidebar.button("Inspect Collection"'),
        ('st.button("🔄 Reset Counter"', 'st.button("Reset Counter"'),
        ('st.button("✅ Confirm Reset")', 'st.button("Confirm Reset")'),
        ('st.button("❌ Cancel")', 'st.button("Cancel")'),
        
        # Subheaders
        ('st.subheader("🎯 Step 1: Select RFP File")', 'st.subheader("Step 1: Select RFP File")'),
        ('st.header(f"✏️ Step 3: Edit Questions & Answers")', 'st.header(f"Step 3: Edit Questions & Answers")'),
        ('st.subheader(f"📄 Current File: {st.session_state.current_file}")', 'st.subheader(f"Current File: {st.session_state.current_file}")'),
        
        # Markdown sections
        ('st.markdown("### 📝 Questions, Answers & Comments")', 'st.markdown("### Questions, Answers & Comments")'),
        ('st.markdown("### 🎛️ Actions")', 'st.markdown("### Actions")'),
        
        # Expanders
        ('st.expander("📁 Document Upload"', 'st.expander("Document Upload"'),
        ('st.sidebar.expander("📄 Processed Files"', 'st.sidebar.expander("Processed Files"'),
        ('st.sidebar.expander("⚠️ Reset RFP Counter"', 'st.sidebar.expander("Reset RFP Counter"'),
        
        # Spinner messages
        ('st.spinner("🤖 Indexing files into vector database...")', 'st.spinner("Indexing files into vector database...")'),
        ('st.spinner("🔍 Inspecting RFP collection...")', 'st.spinner("Inspecting RFP collection...")'),
        ('st.spinner("🔍 Parsing RFP file...")', 'st.spinner("Parsing RFP file...")'),
        
        # Status messages
        ('st.sidebar.success(f"✅ Indexed {indexed_count} documents!")', 'st.sidebar.success(f"Indexed {indexed_count} documents!")'),
        ('st.sidebar.info("📝 No cleanup needed")', 'st.sidebar.info("No cleanup needed")'),
        ('st.info("💡 Use the **📤 Upload New RFPs** section', 'st.info("Use the **Upload New RFPs** section'),
        
        # Error messages  
        ('st.error("❌ ReAct retriever components not available")', 'st.error("ReAct retriever components not available")'),
        ('st.error("❌ Production mode requires OpenAI API key', 'st.error("Production mode requires OpenAI API key'),
        ('st.warning(f"⚠️ Could not process question', 'st.warning(f"Could not process question'),
        ('st.warning(f"⚠️ RFP saved but indexing failed', 'st.warning(f"RFP saved but indexing failed'),
        ('st.error("⚠️ Please enter the submitter name', 'st.error("Please enter the submitter name'),
        ('st.sidebar.error(f"⚠️ RFP Stats Error', 'st.sidebar.error(f"RFP Stats Error'),
        
        # AI mode labels
        ('"🛠️ Development (Free Ollama)"', '"Development (Free Ollama)"'),
        ('"� Production (OpenAI)"', '"Production (OpenAI)"'),
        ('st.sidebar.info("🛠️ Development Mode Only', 'st.sidebar.info("Development Mode Only'),
        
        # Dynamic content
        ('mode_emoji = "🛠️" if current_mode == "dev" else "🚀"', 'mode_label = "Development" if current_mode == "dev" else "Production"'),
        ('f"{mode_emoji} Pre-complete"', 'f"Pre-complete ({mode_label})"'),
        
        # Progress messages
        ('status_text.text(f"🤖 ReAct {mode.upper()} processing', 'status_text.text(f"ReAct {mode.upper()} processing'),
        ('status_text.text(f"✅ ReAct {mode.upper()} pre-completion finished!")', 'status_text.text(f"ReAct {mode.upper()} pre-completion finished!")'),
        ('st.error(f"❌ ReAct {mode.upper()} pre-completion failed', 'st.error(f"ReAct {mode.upper()} pre-completion failed'),
        
        # Statistics
        ('"🧹 Cleanup:', '"Cleanup:'),
        ('"✅ Enabled"', '"Enabled"'),
        ('"❌ Disabled"', '"Disabled"'),
        (' **Statistics:**', '**Statistics:**'),
        (' **Archived as**:', '**Archived as**:'),
        
        # Page config
        ('st.set_page_config(page_title="RFP Manager", page_icon="📋", layout="wide")', 'st.set_page_config(page_title="RFP Manager", layout="wide")'),
    ]
    
    # Apply all replacements
    original_content = content
    for old_text, new_text in replacements:
        content = content.replace(old_text, new_text)
    
    # Write back if changes were made
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Emojis removed from {file_path}")
        return True
    else:
        print(f"No changes made to {file_path}")
        return False

if __name__ == "__main__":
    ui_file = Path("ui/rfp_manager.py")
    if ui_file.exists():
        remove_emojis_from_file(ui_file)
    else:
        print(f"File {ui_file} not found")
