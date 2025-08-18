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
# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

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
    from qdrant.indexer import index_internal_data_with_source, index_completed_rfp
    from qdrant.rfp_tracker import get_rfp_tracker
except Exception as e:
    index_internal_data_with_source = None
    index_completed_rfp = None
    get_rfp_tracker = None

# Import retriever for pre-completion
try:
    from qdrant.retriever import QdrantRetriever
    from qdrant.client import get_qdrant_client, setup_collections_dynamic, INTERNAL_COLLECTION, RFP_QA_COLLECTION
    from qdrant.react_retriever import ReactRFPRetriever
except Exception as e:
    QdrantRetriever = None
    ReactRFPRetriever = None
    get_qdrant_client = None
    setup_collections_dynamic = None


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


def submit_completed_rfp(current_file, completed_df, submitter_name=None):
    """Submit completed RFP: save to Excel, index in vector DB, and move to completed folder"""
    try:
        source_path = project_root / "data" / "new_RFPs" / current_file
        completed_folder = project_root / "data" / "completed_RFPs"
        outputs_folder = project_root / "outputs"
        completed_folder.mkdir(parents=True, exist_ok=True)
        outputs_folder.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp to filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_parts = current_file.rsplit('.', 1)
        new_filename = f"{name_parts[0]}_completed_{timestamp}.{name_parts[1]}"
        
        # Create Excel filename for completed RFP
        excel_filename = f"rfp_completed_{name_parts[0]}_{timestamp}.xlsx"
        excel_filepath = outputs_folder / excel_filename
        
        # Save completed RFP to Excel
        completed_df.to_excel(excel_filepath, index=False)
        
        # Index completed RFP in vector database
        indexed_count = 0
        if index_completed_rfp is not None:
            try:
                # Prepare source information with submitter name
                rfp_source_info = {
                    "client_name": "Unknown",  # Could be extracted from filename
                    "project": name_parts[0],
                    "completion_date": datetime.now().strftime("%Y-%m-%d"),
                    "completion_timestamp": timestamp,
                    "original_filename": current_file,
                    "completed_by": submitter_name or "UI_User",
                    "submission_method": "streamlit_interface"
                }
                
                # Index the completed RFP Q&A pairs
                indexed_count = index_completed_rfp(str(excel_filepath), rfp_source_info)
                
            except Exception as e:
                st.warning(f"⚠️ RFP saved but indexing failed: {e}")
        
        # Move original PDF to completed folder
        dest_path = completed_folder / new_filename
        if source_path.exists():
            shutil.move(str(source_path), str(dest_path))
            return True, new_filename, excel_filename, indexed_count
        else:
            return False, "Source file not found", None, 0
            
    except Exception as e:
        return False, str(e), None, 0


def parse_selected_rfp(selected_file):
    """Parse the selected RFP file and return questions"""
    rfp_path = project_root / "data" / "new_RFPs" / selected_file
    try:
        questions, excel_path = extract_from_pdf(str(rfp_path))
        
        # Convert to DataFrame with required columns including validator
        df_data = []
        for q in questions:
            df_data.append({
                'Questions': q.get('question_text', ''),
                'Answer': '',  # Empty initially
                'Comments': '',  # Empty initially
                'Validator Name': ''  # Empty initially
            })
        
        return pd.DataFrame(df_data)
    except Exception as e:
        st.error(f"Error parsing RFP: {e}")
        return None


def pre_complete_rfp(current_df, mode="dev"):
    """Pre-complete RFP answers using ReAct-based AI retriever"""
    if ReactRFPRetriever is None or get_qdrant_client is None:
        st.error("❌ ReAct retriever components not available")
        return current_df
    
    try:
        # Initialize ReAct retriever only once per session (with mode check)
        session_key = f'react_retriever_{mode}'
        if session_key not in st.session_state or st.session_state[session_key] is None:
            with st.spinner(f"🔧 Setting up ReAct AI system in {mode.upper()} mode..."):
                setup_collections_dynamic()
                client = get_qdrant_client()
                
                # Check if production mode is possible
                if mode == "prod" and not settings.OPENAI_API_KEY:
                    st.error("❌ Production mode requires OpenAI API key. Set OPENAI_API_KEY in .env file.")
                    return current_df
                
                react_retriever = ReactRFPRetriever(client=client, mode=mode)
                
                # Store in session state
                st.session_state[session_key] = react_retriever
        else:
            react_retriever = st.session_state[session_key]
        
        # Process each question
        completed_df = current_df.copy()
        total_questions = len(completed_df)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, row in completed_df.iterrows():
            question = row['Questions']
            if not question or len(question.strip()) < 10:
                continue
            
            # Update progress
            progress = (idx + 1) / total_questions
            progress_bar.progress(progress)
            status_text.text(f"🤖 ReAct {mode.upper()} processing question {idx + 1}/{total_questions}: {question[:50]}...")
            
            try:
                # Use ReAct to answer the question
                result = react_retriever.answer_rfp_question(question)
                
                # Extract the answer
                ai_answer = result.get('answer', 'No answer generated')
                
                # Update the Comments column with the AI answer (include mode info)
                mode_info = f"ReAct AI ({mode.upper()}, {result.get('model', 'unknown')}): {ai_answer}"
                if row['Comments']:
                    completed_df.at[idx, 'Comments'] = f"{row['Comments']} | {mode_info}"
                else:
                    completed_df.at[idx, 'Comments'] = mode_info
                
                # If the answer looks like a Yes/No response, also fill the Answer column
                answer_lower = ai_answer.lower()
                if any(word in answer_lower for word in ['yes', 'oui', 'available', 'supported', 'compliant', 'we have', 'we do', 'we provide']):
                    if not any(neg in answer_lower for neg in ['no', 'not', 'unavailable', 'unsupported', 'we do not', 'we don\'t']):
                        completed_df.at[idx, 'Answer'] = 'Yes'
                elif any(word in answer_lower for word in ['no', 'non', 'unavailable', 'not supported', 'not compliant', 'we do not', 'we don\'t']):
                    completed_df.at[idx, 'Answer'] = 'No'
                
            except Exception as e:
                st.warning(f"⚠️ Could not process question {idx + 1} with ReAct: {e}")
                continue
        
        progress_bar.progress(1.0)
        status_text.text(f"✅ ReAct {mode.upper()} pre-completion finished!")
        
        return completed_df
        
    except Exception as e:
        st.error(f"❌ ReAct {mode.upper()} pre-completion failed: {e}")
        return current_df


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
    
    # RFP Statistics Section
    st.sidebar.header("📊 RFP Statistics")
    
    if get_rfp_tracker:
        try:
            tracker = get_rfp_tracker()
            stats = tracker.get_stats()
            
            col1, col2 = st.sidebar.columns(2)
            with col1:
                st.metric("Current RFP #", stats['current_rfp_number'])
                st.metric("Total Processed", stats['total_rfps_processed'])
            
            with col2:
                cleanup_status = "✅ On" if stats['cleanup_enabled'] else "❌ Off"
                st.metric("Auto Cleanup", cleanup_status)
                st.metric("Max Age Diff", stats['max_age_difference'])
            
            # Cleanup controls
            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.button("🧹 Force Cleanup", help="Clean old RFP data now"):
                    cleanup_count = tracker.cleanup_old_rfps(force=True)
                    if cleanup_count > 0:
                        st.success(f"Cleaned {cleanup_count} old documents")
                    else:
                        st.info("No old documents found")
            
            with col2:
                if st.button("🔄 Reset Counter", help="Reset RFP counter (use carefully)"):
                    st.session_state.show_reset_dialog = True
            
            # Reset dialog
            if st.session_state.get('show_reset_dialog', False):
                with st.sidebar.expander("⚠️ Reset RFP Counter", expanded=True):
                    new_value = st.number_input("New counter value", min_value=0, value=0)
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ Confirm Reset"):
                            tracker.reset_counter(new_value)
                            st.success(f"Counter reset to {new_value}")
                            st.session_state.show_reset_dialog = False
                            st.rerun()
                    with col2:
                        if st.button("❌ Cancel"):
                            st.session_state.show_reset_dialog = False
                            st.rerun()
            
        except Exception as e:
            st.sidebar.error(f"Error loading RFP stats: {e}")
    else:
        st.sidebar.info("RFP tracking not available")
    
    st.sidebar.markdown("---")
    
    # AI Mode Selection
    st.sidebar.header("🤖 AI Processing Mode")
    
    # Check if OpenAI key is available for production mode
    openai_available = bool(settings.OPENAI_API_KEY)
    
    if openai_available:
        ai_mode = st.sidebar.selectbox(
            "Select AI Mode",
            options=["dev", "prod"],
            format_func=lambda x: {
                "dev": "🛠️ Development (Free Ollama)",
                "prod": "🚀 Production (OpenAI)"
            }[x],
            help="Dev mode uses free local Ollama models. Prod mode uses OpenAI for better quality.",
            key="ai_mode_selector"
        )
    else:
        ai_mode = "dev"
        st.sidebar.info("🛠️ Development Mode Only\n\nSet OPENAI_API_KEY in .env to enable production mode with OpenAI.")
    
    # Store AI mode in session state
    st.session_state.ai_mode = ai_mode
    
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
                ),
                "Validator Name": st.column_config.TextColumn(
                    "Validator Name",
                    width="medium",
                    help="Name of the person who validates this Q&A trio"
                )
            },
            hide_index=True,
            key="rfp_editor"
        )
        
        # Update session state with edited data
        st.session_state.rfp_data = edited_df
        
        # Action buttons
        st.markdown("### 🎛️ Actions")
        
        # First row of buttons
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        
        with col1:
            # Get current AI mode
            current_mode = getattr(st.session_state, 'ai_mode', 'dev')
            mode_emoji = "🛠️" if current_mode == "dev" else "🚀"
            
            if st.button(f"{mode_emoji} Pre-complete", type="primary", help=f"Auto-fill answers using ReAct AI in {current_mode.upper()} mode (Internal Docs + RFP History + Web Search)"):
                with st.spinner(f"🤖 Pre-completing RFP using ReAct AI in {current_mode.upper()} mode..."):
                    completed_df = pre_complete_rfp(edited_df, current_mode)
                    st.session_state.rfp_data = completed_df
                    st.rerun()
        
        with col4:
            if st.button("🔄", help="Reset ReAct retriever cache"):
                # Clear both dev and prod caches
                if 'react_retriever_dev' in st.session_state:
                    del st.session_state.react_retriever_dev
                if 'react_retriever_prod' in st.session_state:
                    del st.session_state.react_retriever_prod
                st.success("ReAct cache cleared!")
                st.rerun()
        
        with col2:
            if st.button("💾 Save to Excel", type="secondary"):
                # Save to outputs folder
                outputs_folder = project_root / "outputs"
                outputs_folder.mkdir(exist_ok=True)
                
                filename = f"rfp_edited_{st.session_state.current_file.replace('.pdf', '.xlsx')}"
                filepath = outputs_folder / filename
                
                edited_df.to_excel(filepath, index=False)
                st.success(f"Saved to: {filepath}")
        
        with col3:
            # Submitter name input
            submitter_name = st.text_input(
                "👤 Submitter Name",
                placeholder="Enter your name",
                help="Name of the person submitting this RFP",
                key="submitter_name_input"
            )
            
            if st.button("✅ Submit & Index", type="primary", help="Complete RFP: Save, Index in Vector DB, and Archive"):
                if not submitter_name or not submitter_name.strip():
                    st.error("⚠️ Please enter the submitter name before submitting")
                else:
                    with st.spinner("🚀 Submitting RFP and indexing in vector database..."):
                        success, result, excel_file, indexed_count = submit_completed_rfp(
                            st.session_state.current_file, 
                            edited_df, 
                            submitter_name.strip()
                        )
                        if success:
                            st.success(f"""
                            ✅ **RFP Successfully Submitted!**
                            
                            👤 **Submitted by**: {submitter_name.strip()}
                            📁 **Archived as**: {result}
                            💾 **Excel saved**: {excel_file}
                            🔗 **Vector DB indexed**: {indexed_count} Q&A pairs
                            
                            The RFP knowledge is now available for future AI pre-completion!
                            """)
                            # Clear session state and refresh
                            if 'rfp_data' in st.session_state:
                                del st.session_state.rfp_data
                            if 'current_file' in st.session_state:
                                del st.session_state.current_file
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to submit RFP: {result}")
        
        # Second row of buttons
        st.markdown("---")
        col4, col5, col6 = st.columns(3)
        
        with col4:
            if st.button("🔄 Reset"):
                # Re-parse the original file
                with st.spinner("Resetting..."):
                    df = parse_selected_rfp(st.session_state.current_file)
                    if df is not None:
                        st.session_state.rfp_data = df
                        st.rerun()
        
        with col5:
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
        
        with col6:
            # Placeholder for future features
            pass
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
        
        ### 🤖 Step 2.5: ReAct AI Pre-completion (NEW!)
        - **Pre-complete**: Click "Pre-complete" to auto-fill answers using ReAct AI
        - **Intelligent Reasoning**: Uses ReAct (Reasoning + Acting) pattern with Ollama
        - **Three Smart Tools**: Automatically chooses between Internal Docs + Past RFPs + Web Search
        - **Context-Aware**: AI decides which tools to use based on question content
        - **Comments Integration**: AI answers are placed in the Comments column for review
        - **Smart Yes/No Detection**: Automatically fills Answer column when appropriate
        
        ### ✏️ Step 3: Edit & Review
        - Modify answers and comments in the interactive table
        - Use dropdown for Yes/No answers
        - Add custom comments as needed
        - **NEW**: Enter validator names for each Q&A trio for accountability
        
        ### 💾 Step 4: Save & Submit
        - **Save to Excel**: Export your work to the outputs folder (draft)
        - **Submit & Index**: Complete workflow with submitter name - saves Excel, indexes Q&A in vector DB with validator info, and archives RFP
        
        ### 🔗 Automatic Vector Database Integration
        - **Smart Indexing**: Each question becomes a searchable vector
        - **Rich Metadata**: Answers, comments, validator names, submitter info, and completion date stored
        - **Future AI Enhancement**: Completed RFPs improve future pre-completion accuracy
        - **Knowledge Accumulation**: Building organizational RFP expertise over time
        - **Accountability**: Track who validated each Q&A and who submitted the RFP
        
        ### 🎯 Key Features:
        - ✅ **Drag & Drop Upload**: Easy PDF file management
        - 🤖 **ReAct AI Pre-completion**: Auto-fill answers using intelligent reasoning (NEW!)
        - 🧠 **Multi-Tool Intelligence**: AI chooses best sources (Docs/RFPs/Web) per question
        - 📝 **Interactive Editing**: Real-time table editing
        - � **Validation & Accountability**: Track validator names and submitter info (NEW!)
        - �📊 **Progress Tracking**: Live statistics and completion percentage
        - 💾 **Excel Export**: Professional output formatting
        - 🔄 **Reset**: Start over with original parsed questions
        """)


if __name__ == "__main__":
    main()
