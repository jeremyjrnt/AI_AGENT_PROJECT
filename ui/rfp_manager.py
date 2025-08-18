#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RFP User Interface - Interactive RFP Question Management
Allows users to:
1. Select RFP PDFs from data/new_RFPs
2. Parse them with rfp_parser
3. View and edit Questions, Answers, and Comments table
"""

import streamlit as st
import pandas as pd
import os
from pathlib import Path
import sys
import shutil

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from parsers.rfp_parser import extract_from_pdf
from qdrant.client import get_qdrant_client, INTERNAL_COLLECTION
from sentence_transformers import SentenceTransformer
import settings
import openai
import importlib.util
import tempfile
from datetime import datetime
import zipfile
import io

# Import pre-filler functions with proper error handling
try:
    spec = importlib.util.spec_from_file_location("pre_filler", project_root / "rfp_filler" / "pre-filler.py")
    pre_filler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pre_filler)
    search_answers_and_contexts = pre_filler.search_answers_and_contexts
    process_rfp_with_ai = pre_filler.process_rfp_with_ai
except Exception as e:
    search_answers_and_contexts = None
    process_rfp_with_ai = None

# Import indexing functions
try:
    from qdrant.indexer import index_internal_data_with_source
except Exception as e:
    index_internal_data_with_source = None


def load_rfp_files():
    """Load available RFP PDF files from data/new_RFPs"""
    rfp_folder = project_root / "data" / "new_RFPs"
    if not rfp_folder.exists():
        rfp_folder.mkdir(parents=True, exist_ok=True)
        return []
    
    pdf_files = list(rfp_folder.glob("*.pdf"))
    return [f.name for f in pdf_files]


def handle_file_upload(uploaded_files):
    """Handle drag and drop file upload to new_RFPs folder"""
    rfp_folder = project_root / "data" / "new_RFPs"
    rfp_folder.mkdir(parents=True, exist_ok=True)
    
    uploaded_count = 0
    for uploaded_file in uploaded_files:
        if uploaded_file.name.lower().endswith('.pdf'):
            file_path = rfp_folder / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            uploaded_count += 1
        else:
            st.warning(f"Skipped {uploaded_file.name} - Only PDF files are accepted")
    
    return uploaded_count


def handle_internal_docs_upload(uploaded_files, source_name):
    """Handle upload and indexing of internal documentation files"""
    if not uploaded_files:
        return 0, []
    
    if index_internal_data_with_source is None:
        st.error("Indexing function not available")
        return 0, []
    
    # Create temporary directory for uploaded files
    temp_dir = project_root / "temp_internal_docs"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        uploaded_count = 0
        processed_files = []
        
        # Save uploaded files to temp directory
        for uploaded_file in uploaded_files:
            file_extension = uploaded_file.name.lower().split('.')[-1]
            
            # Support text and markdown files
            if file_extension in ['txt', 'md', 'markdown']:
                file_path = temp_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                uploaded_count += 1
                processed_files.append(uploaded_file.name)
            
            # Support ZIP files containing text files
            elif file_extension == 'zip':
                try:
                    with zipfile.ZipFile(io.BytesIO(uploaded_file.getbuffer()), 'r') as zip_ref:
                        for file_info in zip_ref.filelist:
                            if file_info.filename.lower().endswith(('.txt', '.md', '.markdown')):
                                # Extract to temp directory
                                zip_ref.extract(file_info, temp_dir)
                                uploaded_count += 1
                                processed_files.append(f"{uploaded_file.name}/{file_info.filename}")
                except Exception as e:
                    st.warning(f"Error processing ZIP file {uploaded_file.name}: {e}")
            else:
                st.warning(f"Skipped {uploaded_file.name} - Only .txt, .md, .markdown and .zip files are supported")
        
        # Index the uploaded files
        if uploaded_count > 0:
            indexed_count = index_internal_data_with_source(
                str(temp_dir), 
                source_name or "uploaded_docs"
            )
            
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            return indexed_count, processed_files
        else:
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            return 0, []
            
    except Exception as e:
        # Clean up temp directory on error
        shutil.rmtree(temp_dir, ignore_errors=True)
        st.error(f"Error during indexing: {e}")
        return 0, []


def parse_selected_rfp(selected_file):
    """Parse the selected RFP file and return questions"""
    rfp_path = project_root / "data" / "new_RFPs" / selected_file
    try:
        questions, excel_path = extract_from_pdf(str(rfp_path))
        
        # Convert to DataFrame with required columns
        df_data = []
        for q in questions:
            df_data.append({
                'Questions': q.get('question_text', ''),
                'Answer': '',  # Empty initially
                'Comments': ''  # Empty initially
            })
        
        return pd.DataFrame(df_data)
    except Exception as e:
        st.error(f"Error parsing RFP: {e}")
        return None


def main():
    st.set_page_config(page_title="RFP Manager", page_icon="📋", layout="wide")
    
    # Header
    st.title("🚀 RFP Question Manager")
    st.markdown("---")
    
    # Sidebar for file management
    st.sidebar.header("📁 RFP File Management")
    
    # File upload section for RFPs
    st.sidebar.subheader("📤 Upload New RFPs")
    uploaded_files = st.sidebar.file_uploader(
        "Drag and drop PDF files here",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload PDF files to add them to the new_RFPs folder",
        key="rfp_uploader"
    )
    
    if uploaded_files:
        uploaded_count = handle_file_upload(uploaded_files)
        if uploaded_count > 0:
            st.sidebar.success(f"✅ Uploaded {uploaded_count} PDF file(s)")
            st.rerun()  # Refresh to show new files
    
    st.sidebar.markdown("---")
    
    # Internal documents upload section
    st.sidebar.header("📚 Knowledge Base Management")
    
    # Source name input
    source_name = st.sidebar.text_input(
        "Source Name",
        value="company_docs",
        help="Name to identify this batch of documents",
        key="source_name_files"
    )
    
    # Individual files uploader
    internal_docs = st.sidebar.file_uploader(
        "Upload documents (.txt, .md, .zip)",
        type=['txt', 'md', 'markdown', 'zip'],
        accept_multiple_files=True,
        help="Upload text files or ZIP archives",
        key="internal_docs_files"
    )
    
    if internal_docs:
        if st.sidebar.button("🔄 Index Files in Vector DB", type="primary", key="index_files"):
            with st.spinner("🤖 Indexing files into vector database..."):
                indexed_count, processed_files = handle_internal_docs_upload(internal_docs, source_name)
                
                if indexed_count > 0:
                    st.sidebar.success(f"✅ Indexed {indexed_count} documents!")
                    
                    with st.sidebar.expander("📄 Processed Files", expanded=True):
                        for file_name in processed_files:
                            st.sidebar.write(f"• {file_name}")
                    
                    st.rerun()
                else:
                    st.sidebar.warning("⚠️ No documents were indexed")
    
    st.sidebar.markdown("---")
    
    # Load available RFP files
    rfp_files = load_rfp_files()
    
    if not rfp_files:
        st.sidebar.warning("No RFP files found in data/new_RFPs/")
        st.sidebar.info("👆 Upload PDF files using the drag & drop area above")
    else:
        # File selection
        st.sidebar.subheader("📋 Select RFP File")
        selected_file = st.sidebar.selectbox(
            "Choose an RFP file:",
            rfp_files,
            help="Select a PDF file from data/new_RFPs to parse and edit"
        )
    
        if st.sidebar.button("📖 Parse RFP", type="primary"):
            with st.spinner("Parsing RFP file..."):
                df = parse_selected_rfp(selected_file)
                
                if df is not None:
                    st.session_state.rfp_data = df
                    st.session_state.current_file = selected_file
                    st.success(f"Successfully parsed {len(df)} questions from {selected_file}")
    
    # Main content area
    if 'rfp_data' in st.session_state:
        st.header(f"📋 Editing: {st.session_state.current_file}")
        
        # Display editable table
        st.subheader("Questions, Answers & Comments")
        
        # Create editable data editor
        edited_df = st.data_editor(
            st.session_state.rfp_data,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "Questions": st.column_config.TextColumn(
                    "Questions",
                    width="large",
                    help="RFP questions extracted from the document"
                ),
                "Answer": st.column_config.SelectboxColumn(
                    "Answer",
                    width="small",
                    options=["", "Yes", "No"],
                    help="Select Yes or No for each question"
                ),
                "Comments": st.column_config.TextColumn(
                    "Comments",
                    width="large",
                    help="Add comments or explanations for each answer"
                )
            },
            hide_index=True,
            key="rfp_editor"
        )
        
        # Update session state with edited data
        st.session_state.rfp_data = edited_df
        
        # Action buttons
        st.markdown("### 🎛️ Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save to Excel", type="primary"):
                # Save to outputs folder
                outputs_folder = project_root / "outputs"
                outputs_folder.mkdir(exist_ok=True)
                
                filename = f"rfp_edited_{st.session_state.current_file.replace('.pdf', '.xlsx')}"
                filepath = outputs_folder / filename
                
                edited_df.to_excel(filepath, index=False)
                st.success(f"Saved to: {filepath}")
        
        with col2:
            if st.button("🔄 Reset"):
                # Re-parse the original file
                with st.spinner("Resetting..."):
                    df = parse_selected_rfp(st.session_state.current_file)
                    if df is not None:
                        st.session_state.rfp_data = df
                        st.rerun()
        
        with col3:
            if st.button("📊 Statistics"):
                total_questions = len(edited_df)
                answered_yes = len(edited_df[edited_df['Answer'] == 'Yes'])
                answered_no = len(edited_df[edited_df['Answer'] == 'No'])
                unanswered = len(edited_df[edited_df['Answer'] == ''])
                
                st.info(f"""
                📈 **Statistics:**
                - Total Questions: {total_questions}
                - Answered 'Yes': {answered_yes}
                - Answered 'No': {answered_no}
                - Unanswered: {unanswered}
                - Progress: {((answered_yes + answered_no) / total_questions * 100):.1f}%
                """)
    else:
        # Welcome message
        st.info("👆 Select an RFP file from the sidebar to get started")
        
        # Show instructions
        st.markdown("""
        ## 🚀 How to use this RFP Manager:
        
        ### 📤 Step 1: Upload RFP Files
        - **Drag & Drop**: Use the sidebar file uploader to drag PDF files
        - **Manual**: Add PDF files directly to the `data/new_RFPs/` folder
        
        ### 📚 Step 1.5: Build Knowledge Base (Optional)
        - **Upload Internal Docs**: Use the "Knowledge Base Management" section
        - **Supported Formats**: .txt, .md, .markdown files or .zip archives
        - **Auto-Index**: Documents are automatically indexed in vector database
        
        ### 📋 Step 2: Process RFP
        1. **Select**: Choose an RFP PDF from the dropdown
        2. **Parse**: Click "Parse RFP" to extract questions
        
        ### ✏️ Step 3: Edit & Review
        - Modify answers and comments in the interactive table
        - Use dropdown for Yes/No answers
        - Add custom comments as needed
        
        ### 💾 Step 4: Save
        - **Save to Excel**: Export your work to the outputs folder
        
        ### 🎯 Key Features:
        - ✅ **Drag & Drop Upload**: Easy PDF file management
        - 📝 **Interactive Editing**: Real-time table editing
        - 📊 **Progress Tracking**: Live statistics and completion percentage
        - 💾 **Excel Export**: Professional output formatting
        - 🔄 **Reset**: Start over with original parsed questions
        """)


if __name__ == "__main__":
    main()
