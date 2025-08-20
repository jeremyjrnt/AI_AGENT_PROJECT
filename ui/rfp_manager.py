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
import base64

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from parsers.rfp_parser import extract_rfp_questions
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
                st.warning(f"RFP saved but indexing failed: {e}")
        
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
        # Use the new RFP parser function without file management
        original_pdf_path, questions_list = extract_rfp_questions(str(rfp_path), manage_files=False)
        
        # Convert questions list to DataFrame directly
        df_data = []
        for question in questions_list:
            df_data.append({
                'Questions': str(question),
                'Answer': '',  # Always start empty for user input
                'Comments': '',  # Always start empty for user input
                'Validator Name': ''  # Empty initially
            })
        
        # Create DataFrame with explicit string types
        df = pd.DataFrame(df_data)
        # Ensure all columns are explicitly string type
        df = df.astype({
            'Questions': 'string',
            'Answer': 'string', 
            'Comments': 'string',
            'Validator Name': 'string'
        })
        
        return df
    except Exception as e:
        st.error(f"Error parsing RFP: {e}")
        return None


def show_pdf_viewer(pdf_file_path):
    """Display PDF file in the Streamlit interface"""
    try:
        with open(pdf_file_path, "rb") as f:
            pdf_bytes = f.read()
            
        # Convert to base64 for embedding
        pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
        
        # Create PDF viewer using iframe
        pdf_display = f"""
        <iframe src="data:application/pdf;base64,{pdf_base64}" 
                width="100%" 
                height="600px" 
                style="border: 1px solid #ccc; border-radius: 5px;">
        </iframe>
        """
        
        st.markdown(pdf_display, unsafe_allow_html=True)
        
        # Add file info
        file_size = len(pdf_bytes) / (1024 * 1024)  # Convert to MB
        st.info(f"**File**: {Path(pdf_file_path).name} | **Size**: {file_size:.2f} MB")
        
        return True
    except Exception as e:
        st.error(f"Could not display PDF: {e}")
        st.info("💡 The file might be corrupted or not a valid PDF format")
        return False


def pre_complete_rfp(current_df, mode="dev"):
    """Pre-complete RFP answers using ReAct-based AI retriever"""
    if ReactRFPRetriever is None or get_qdrant_client is None:
        st.error("ReAct retriever components not available")
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
                    st.error("Production mode requires OpenAI API key. Set OPENAI_API_KEY in .env file.")
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
            status_text.text(f"ReAct {mode.upper()} processing question {idx + 1}/{total_questions}: {question[:50]}...")
            
            try:
                # Use ReAct to answer the question with structured Yes/No format
                result = react_retriever.answer_rfp_question(question)
                
                # Extract structured answer and comments
                binary_answer = result.get('answer', 'No')  # Yes/No
                detailed_comments = result.get('comments', 'No detailed explanation provided')
                
                # Fill the Answer column with Yes/No
                completed_df.at[idx, 'Answer'] = binary_answer
                
                # Fill the Comments column with detailed explanation
                mode_info = f"ReAct AI ({mode.upper()}, {result.get('model', 'unknown')})"
                if row['Comments'] and row['Comments'].strip():
                    # Preserve existing comments and add AI explanation
                    completed_df.at[idx, 'Comments'] = f"{row['Comments'].strip()} | {mode_info}: {detailed_comments}"
                else:
                    # Use only AI explanation
                    completed_df.at[idx, 'Comments'] = f"{mode_info}: {detailed_comments}"
                
            except Exception as e:
                st.warning(f"Could not process question {idx + 1} with ReAct: {e}")
                # Set default values for errors
                completed_df.at[idx, 'Answer'] = 'No'
                completed_df.at[idx, 'Comments'] = f"Error: Could not process with ReAct AI - {str(e)}"
                continue
        
        progress_bar.progress(1.0)
        status_text.text(f"ReAct {mode.upper()} pre-completion finished!")
        
        return completed_df
        
    except Exception as e:
        st.error(f"ReAct {mode.upper()} pre-completion failed: {e}")
        return current_df


def main():
    st.set_page_config(
        page_title="RFP Manager", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS for professional styling
    st.markdown("""
    <style>
    .main > div {
        padding-top: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #f8f9fa;
        padding: 6px;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        font-weight: 500;
        border-radius: 6px;
        color: #495057;
        border: none;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e9ecef;
        color: #495057;
    }
    .stTabs [aria-selected="true"] {
        background-color: #007bff !important;
        color: white !important;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
    }
    .stButton > button {
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create main tabs
    tab1, tab2 = st.tabs(["RFP Manager", "User Guide"])
    
    with tab1:
        show_rfp_manager()
    
    with tab2:
        show_how_to_use()

def show_how_to_use():
    """How to Use guide interface"""
    st.title("How to Use this RFP Manager")
    st.markdown("---")
    
    st.markdown("""
    ## Complete Guide to RFP Management System
    
    ### Step 1: Upload RFP Files
    - **Drag & Drop**: Use the sidebar file uploader to drag PDF files
    - **Manual**: Add PDF files directly to the `data/new_RFPs/` folder
    - **Supported**: Only PDF format is supported for RFP processing
    
    ### Step 1.5: Build Knowledge Base (Optional)
    - **Upload Internal Docs**: Use the "Knowledge Base Management" section
    - **Supported Formats**: .txt, .md, .markdown files or .zip archives
    - **Auto-Index**: Documents are automatically indexed in vector database
    - **Smart Search**: Indexed documents will be used for AI-powered answer suggestions
    
    ### Step 2: Process RFP
    1. **Select**: Choose an RFP PDF from the main interface dropdown
    2. **Parse**: Click "Parse RFP" to extract questions automatically
    3. **Review**: Check the extracted questions for accuracy
    
    ### Step 2.5: ReAct AI Pre-completion (NEW!)
    - **Pre-complete**: Click "Pre-complete" to auto-fill answers using ReAct AI
    - **Intelligent Reasoning**: Uses ReAct (Reasoning + Acting) pattern with Ollama
    - **Three Smart Tools**: Automatically chooses between Internal Docs + Past RFPs + Web Search
    - **Context-Aware**: AI decides which tools to use based on question content
    - **Comments Integration**: AI answers are placed in the Comments column for review
    - **Smart Yes/No Detection**: Automatically fills Answer column when appropriate
    
    ### Step 3: Edit & Review
    - **Interactive Table**: Modify answers and comments directly in the interface
    - **Dropdown Answers**: Use dropdown for Yes/No answers
    - **Custom Comments**: Add detailed explanations and context
    - **Validator Names**: Enter validator names for each Q&A trio for accountability
    - **Real-time Editing**: Changes are saved automatically as you type
    
    ### Step 4: Save & Submit
    - **Save to Excel**: Export your work to the outputs folder (draft version)
    - **Submit & Index**: Complete workflow with submitter name:
      - Saves Excel file with all answers and metadata
      - Indexes Q&A pairs in vector database with validator info
      - Archives RFP for future reference
      - Triggers automatic cleanup of old RFPs if needed
    
    ### Automatic Vector Database Integration
    - **Smart Indexing**: Each question becomes a searchable vector in Qdrant
    - **Rich Metadata**: Answers, comments, validator names, submitter info, and completion date stored
    - **Future AI Enhancement**: Completed RFPs improve future pre-completion accuracy
    - **Knowledge Accumulation**: Building organizational RFP expertise over time
    - **Accountability**: Full traceability of who validated each Q&A and who submitted the RFP
    - **Age Management**: Automatic cleanup of old RFPs (configurable, default 20 RFPs)
    
    ### Key Features Overview:
    """)
    
    # Feature cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Core Features
        - **Drag & Drop Upload**: Easy PDF file management
        - **Smart Parsing**: Automatic question extraction from PDFs
        - **Interactive Editing**: Real-time table editing with validation
        - **Excel Export**: Professional output formatting
        - **Reset Function**: Start over with original parsed questions
        """)
        
    with col2:
        st.markdown("""
        #### AI Features
        - **ReAct AI Pre-completion**: Auto-fill answers using intelligent reasoning
        - **Multi-Tool Intelligence**: AI chooses best sources per question
        - **Progress Tracking**: Live statistics and completion percentage
        - **Validation & Accountability**: Track validator names and submitter info
        - **Vector Database**: Smart indexing for future improvements
        """)
    
    st.markdown("---")
    
    # Technical details
    with st.expander(" Technical Details", expanded=False):
        st.markdown("""
        ### System Architecture
        - **Frontend**: Streamlit web interface
        - **Vector Database**: Qdrant for semantic search
        - **AI Models**: Sentence Transformers + Ollama/OpenAI
        - **File Processing**: PDF parsing with question extraction
        - **Data Storage**: Excel export + vector indexing
        
        ### RFP Numbering System
        - **Automatic Numbering**: Each processed RFP gets a unique sequential number
        - **Metadata Tracking**: RFP number stored in all generated documents
        - **Age-based Cleanup**: Automatically removes old RFP data (default: 20 RFPs difference)
        - **State Management**: Persistent tracking across sessions
        
        ### AI Processing Modes
        - **Development Mode**: Uses free Ollama models locally
        - **Production Mode**: Uses OpenAI API for better accuracy (requires API key)
        - **Hybrid Approach**: Combines multiple data sources for best results
        """)
    
    # Tips and best practices
    with st.expander("Tips & Best Practices", expanded=False):
        st.markdown("""
        ### For Best Results:
        1. **Upload Quality PDFs**: Clear, text-based PDFs work best
        2. **Build Knowledge Base**: Upload internal documentation before processing
        3. **Review AI Suggestions**: Always validate AI-generated answers
        4. **Use Validator Names**: Track who validated each answer for accountability
        5. **Regular Cleanup**: Monitor RFP statistics and cleanup old data periodically
        
        ### Troubleshooting:
        - **No Questions Found**: Check PDF quality and format
        - **AI Not Working**: Verify Ollama is running (dev mode) or API keys (prod mode)
        - **Slow Performance**: Consider reducing batch size or upgrading hardware
        - **Database Issues**: Use the cleanup tools in sidebar statistics section
        """)

def show_rfp_manager():
    """Main RFP Manager interface"""
    
    # Header with RFP counter
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("RFP Question Manager")
    with col2:
        if get_rfp_tracker is not None:
            try:
                tracker = get_rfp_tracker()
                current_rfp = tracker.get_current_rfp_number()
                next_rfp = current_rfp + 1
                st.metric("Next RFP #", next_rfp, help="Number that will be assigned to the next processed RFP")
            except:
                pass
    
    st.markdown("---")
    
    # Sidebar for file management
    st.sidebar.header("File Management")
    
    # File upload section for RFPs
    st.sidebar.subheader("Upload New RFPs")
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
            st.sidebar.success(f"Uploaded {uploaded_count} PDF file(s)")
            st.rerun()  # Refresh to show new files
    
    st.sidebar.markdown("---")
    
    # Internal documents upload section
    st.sidebar.header("Knowledge Base Management")
    
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
        if st.sidebar.button("Index Files in Vector DB", type="primary", key="index_files"):
            with st.spinner("Indexing files into vector database..."):
                indexed_count, processed_files = handle_internal_docs_upload(internal_docs, source_name)
                
                if indexed_count > 0:
                    st.sidebar.success(f"Indexed {indexed_count} documents!")
                    
                    with st.sidebar.expander("Processed Files", expanded=True):
                        for file_name in processed_files:
                            st.sidebar.write(f"• {file_name}")
                    
                    st.rerun()
                else:
                    st.sidebar.warning("⚠️ No documents were indexed")
    
    st.sidebar.markdown("---")
    
    # RFP Statistics Section
    st.sidebar.header("RFP Statistics")
    
    if get_rfp_tracker:
        try:
            tracker = get_rfp_tracker()
            stats = tracker.get_stats()
            
            col1, col2 = st.sidebar.columns(2)
            with col1:
                st.metric("Current RFP #", stats['current_rfp_number'])
                st.metric("Total Processed", stats['total_rfps_processed'])
            
            with col2:
                cleanup_status = "On" if stats['cleanup_enabled'] else "Off"
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
                if st.button("Reset Counter", help="Reset RFP counter (use carefully)"):
                    st.session_state.show_reset_dialog = True
            
            # Reset dialog
            if st.session_state.get('show_reset_dialog', False):
                with st.sidebar.expander("Reset RFP Counter", expanded=True):
                    new_value = st.number_input("New counter value", min_value=0, value=0)
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Confirm Reset"):
                            tracker.reset_counter(new_value)
                            st.success(f"Counter reset to {new_value}")
                            st.session_state.show_reset_dialog = False
                            st.rerun()
                    with col2:
                        if st.button("Cancel"):
                            st.session_state.show_reset_dialog = False
                            st.rerun()
            
        except Exception as e:
            st.sidebar.error(f"Error loading RFP stats: {e}")
    else:
        st.sidebar.info("RFP tracking not available")
    
    st.sidebar.markdown("---")
    
    # RFP Statistics Section
    if get_rfp_tracker is not None:
        st.sidebar.header("RFP Statistics")
        
        try:
            tracker = get_rfp_tracker()
            stats = tracker.get_stats()
            
            # Display key metrics
            col1, col2 = st.sidebar.columns(2)
            with col1:
                st.metric("Current RFP", stats['current_rfp_number'])
            with col2:
                st.metric("Total Processed", stats['total_rfps_processed'])
            
            # Cleanup settings
            st.sidebar.write(f"Cleanup: {'Enabled' if stats['cleanup_enabled'] else 'Disabled'}")
            st.sidebar.write(f"Max Age: {stats['max_age_difference']} RFPs")
            
            # Management buttons
            if st.sidebar.button("Inspect Collection", help="View RFP age distribution"):
                with st.spinner("Inspecting RFP collection..."):
                    # This could be expanded to show results in main area
                    st.sidebar.success("Check terminal for detailed analysis")
            
            if st.sidebar.button("Force Cleanup", help="Remove old RFP documents"):
                with st.spinner("🧹 Cleaning up old RFPs..."):
                    cleanup_count = tracker.cleanup_old_rfps(force=True)
                    if cleanup_count > 0:
                        st.sidebar.success(f"Cleaned {cleanup_count} documents")
                    else:
                        st.sidebar.info("No cleanup needed")
                    st.rerun()
            
        except Exception as e:
            st.sidebar.error(f"RFP Stats Error: {e}")
    
    st.sidebar.markdown("---")
    
    # Load available RFP files
    rfp_files = load_rfp_files()
    
    if not rfp_files:
        st.sidebar.warning("No RFP files found in data/new_RFPs/")
        st.sidebar.info("Upload PDF files using the drag & drop area above")
    else:
        # No files available message will be shown in main body
        pass
    
    # Main content area - RFP Selection and Processing
    
    # Step 1: RFP File Selection
    if not rfp_files:
        st.warning("No RFP files found. Please upload PDF files using the sidebar.")
        st.info("Use the **Upload New RFPs** section in the sidebar to add PDF files.")
    else:
        # RFP File Selection in main body
        st.subheader("Select RFP File")
        selected_file = st.selectbox(
            "Choose an RFP PDF file to process:",
            options=rfp_files,
            help="Select a PDF file from the uploaded RFPs.",
            key="main_rfp_selector"
        )
        
        # Step 2: PDF Preview (Auto-display when file selected)
        if selected_file:
            st.markdown("---")
            st.subheader("PDF Preview")
            
            rfp_path = project_root / "data" / "new_RFPs" / selected_file
            
            # Automatically show PDF preview
            with st.spinner("Loading PDF preview..."):
                show_pdf_viewer(str(rfp_path))
            
            st.markdown("---")
            st.subheader("Extract Questions")
            
            # Parse button centered
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("Extract questions from RFP", type="primary", use_container_width=True):
                    with st.spinner("Parsing RFP file..."):
                        df = parse_selected_rfp(selected_file)
                        
                        if df is not None:
                            st.session_state.rfp_data = df
                            st.session_state.current_file = selected_file
                            st.success(f"Successfully parsed {len(df)} questions from {selected_file}")
                            st.rerun()  # Refresh to show editing interface
                        else:
                            st.error("Failed to parse the RFP file. Please check the file format.")
        
        st.markdown("---")
    
    # Step 4: Editing Interface (only shown after parsing)
    if 'rfp_data' in st.session_state:
        st.header(f"Edit Questions & Answers")
        st.subheader(f"Current File: {st.session_state.current_file}")
        
        # Display editable table
        st.markdown("### Questions, Answers & Comments")
        
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
        
        # Simplified Action buttons
        st.markdown("### Actions")
        
        # Three main action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Pre-complete", type="primary", use_container_width=True, help="Auto-fill answers using AI with live progress"):
                if ReactRFPRetriever is None or get_qdrant_client is None:
                    st.error("ReAct retriever components not available")
                else:
                    # Check if OpenAI key is available
                    if not settings.OPENAI_API_KEY:
                        st.error("OpenAI API key required. Please set OPENAI_API_KEY in .env file.")
                    else:
                        # Process all questions with AI with live updates (production mode only)
                        with st.spinner(f"Processing {len(edited_df)} questions with AI..."):
                            try:
                                # Initialize ReAct retriever in production mode
                                setup_collections_dynamic()
                                client = get_qdrant_client()
                                react_retriever = ReactRFPRetriever(client=client, mode="prod")
                                
                                # Create container for live updates
                                progress_container = st.container()
                                table_container = st.container()
                                
                                with progress_container:
                                    progress_bar = st.progress(0)
                                    status_text = st.empty()
                                
                                # Process each question with live table updates
                                processed_df = edited_df.copy()
                                
                                for idx, row in processed_df.iterrows():
                                    question = row['Questions']
                                    if not question or len(str(question).strip()) < 10:
                                        continue
                                    
                                    # Update progress
                                    progress = (idx + 1) / len(processed_df)
                                    progress_bar.progress(progress)
                                    status_text.text(f"Processing question {idx + 1}/{len(processed_df)}: {str(question)[:60]}...")
                                    
                                    try:
                                        # Use ReAct to answer the question with structured Yes/No format
                                        result = react_retriever.answer_rfp_question(str(question))
                                        
                                        # Extract structured answer and comments
                                        binary_answer = result.get('answer', 'No')  # Yes/No
                                        detailed_comments = result.get('comments', 'No detailed explanation provided')
                                        
                                        # Fill the Answer column with Yes/No
                                        processed_df.at[idx, 'Answer'] = binary_answer
                                        
                                        # Fill the Comments column with detailed explanation
                                        if row['Comments'] and str(row['Comments']).strip() and str(row['Comments']).strip() != 'nan':
                                            # Preserve existing comments and add AI explanation
                                            processed_df.at[idx, 'Comments'] = f"{str(row['Comments']).strip()} | AI: {detailed_comments}"
                                        else:
                                            # Use only AI explanation
                                            processed_df.at[idx, 'Comments'] = f"AI: {detailed_comments}"
                                        
                                        # Show live update of the table with current progress
                                        with table_container:
                                            st.subheader(f"📝 Live Progress - Question {idx + 1}/{len(processed_df)} completed")
                                            # Show only processed rows so far for performance
                                            display_df = processed_df.iloc[:idx + 1].copy()
                                            # Add status indicator
                                            for i in range(len(display_df)):
                                                if i <= idx:
                                                    # Show completed rows with check mark
                                                    if display_df.iloc[i]['Answer']:
                                                        display_df.iloc[i, 0] = f"{display_df.iloc[i, 0]}"
                                            
                                            st.dataframe(
                                                display_df[['Questions', 'Answer', 'Comments']],
                                                use_container_width=True,
                                                hide_index=True
                                            )
                                        
                                        # Small delay to make updates visible
                                        import time
                                        time.sleep(0.5)
                                        
                                    except Exception as e:
                                        st.warning(f"Could not process question {idx + 1} with AI: {e}")
                                        # Set default values for errors
                                        processed_df.at[idx, 'Answer'] = 'No'
                                        processed_df.at[idx, 'Comments'] = f"Error: Could not process with AI - {str(e)[:100]}"
                                        
                                        # Still show the update even for errors
                                        with table_container:
                                            st.subheader(f"⚠️ Question {idx + 1}/{len(processed_df)} - Processing Error")
                                            display_df = processed_df.iloc[:idx + 1].copy()
                                            st.dataframe(
                                                display_df[['Questions', 'Answer', 'Comments']],
                                                use_container_width=True,
                                                hide_index=True
                                            )
                                        
                                        time.sleep(0.5)
                                        continue
                                
                                progress_bar.progress(1.0)
                                status_text.text(f"AI processing complete! All {len(processed_df)} questions processed.")
                                
                                # Final table display
                                with table_container:
                                    st.subheader("Final Results - All Questions Completed")
                                    st.dataframe(
                                        processed_df[['Questions', 'Answer', 'Comments']],
                                        use_container_width=True,
                                        hide_index=True
                                    )
                                
                                # Update the session state with processed data
                                st.session_state.rfp_data = processed_df
                                st.success(f"Successfully processed {len(processed_df)} questions with AI!")
                                
                                # Auto-refresh after 2 seconds to show in main editor
                                import time
                                time.sleep(2)
                                st.rerun()  # Refresh to show updated data
                                
                            except Exception as e:
                                st.error(f"AI processing failed: {e}")
        
        with col2:
            if st.button("Save Excel", type="secondary", use_container_width=True):
                # Save to outputs folder
                outputs_folder = project_root / "outputs"
                outputs_folder.mkdir(exist_ok=True)
                
                filename = f"rfp_edited_{st.session_state.current_file.replace('.pdf', '.xlsx')}"
                filepath = outputs_folder / filename
                
                edited_df.to_excel(filepath, index=False)
                st.success(f"Saved to: {filepath}")
        
        with col3:
            # Submit button with validator name
            validator_name = st.text_input(
                "Validator Name",
                placeholder="Enter validator name",
                help="Name of the person validating this RFP",
                key="validator_name_input"
            )
            
            if st.button("Submit", type="primary", use_container_width=True, help="Complete RFP: Save, Index in Vector DB, and Archive"):
                if not validator_name or not validator_name.strip():
                    st.error("Please enter the validator name before submitting")
                else:
                    with st.spinner("Submitting RFP and indexing in vector database..."):
                        success, result, excel_file, indexed_count = submit_completed_rfp(
                            st.session_state.current_file, 
                            edited_df, 
                            validator_name.strip()
                        )
                        if success:
                            st.success(f"""
                            **RFP Successfully Submitted!**
                            
                            👤 **Validated by**: {validator_name.strip()}
                            📁 **Archived as**: {result}
                            💾 **Excel saved**: {excel_file}
                            🔗 **Vector DB indexed**: {indexed_count} Q&A pairs
                            
                            The RFP knowledge is now available for future AI processing!
                            """)
                            # Clear session state and refresh
                            if 'rfp_data' in st.session_state:
                                del st.session_state.rfp_data
                            if 'current_file' in st.session_state:
                                del st.session_state.current_file
                            st.rerun()
                        else:
                            st.error(f"Failed to submit RFP: {result}")
        
        # Optional secondary actions
        st.markdown("---")
        col4, col5, col6 = st.columns(3)
        
        with col4:
            if st.button("Reset", help="Clear Answer and Comments columns"):
                # Clear Answer and Comments columns only
                with st.spinner("Clearing Answer and Comments..."):
                    reset_df = edited_df.copy()
                    reset_df['Answer'] = ''
                    reset_df['Comments'] = ''
                    st.session_state.rfp_data = reset_df
                    st.success("Answer and Comments columns cleared!")
                    st.rerun()
        
        with col5:
            if st.button("Statistics", help="Show progress statistics"):
                total_questions = len(edited_df)
                answered_yes = len(edited_df[edited_df['Answer'] == 'Yes'])
                answered_no = len(edited_df[edited_df['Answer'] == 'No'])
                unanswered = len(edited_df[edited_df['Answer'] == ''])
                
                st.info(f"""
                **Statistics:**
                - Total Questions: {total_questions}
                - Answered 'Yes': {answered_yes}
                - Answered 'No': {answered_no}
                - Unanswered: {unanswered}
                - Progress: {((answered_yes + answered_no) / total_questions * 100):.1f}%
                """)
        
        with col6:
            # Empty column (removed Clear Cache button)
            pass
        # Welcome message when no RFP is loaded
    if 'rfp_data' not in st.session_state:
        st.info("Select an RFP file from above to get started")


if __name__ == "__main__":
    main()
