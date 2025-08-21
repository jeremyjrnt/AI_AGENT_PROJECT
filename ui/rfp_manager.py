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

# Import agents instead of direct functions
from agent.rfp_parser_agent import RFPParserAgent
from agent.rfp_completion_agent import RFPCompletionAgent

from qdrant.client import get_qdrant_client, INTERNAL_COLLECTION
from qdrant.indexer import upsert_rfp
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
from qdrant.indexer import upsert_rfp
from qdrant.rfp_tracker import get_rfp_tracker

from qdrant.client import get_qdrant_client, setup_collections_dynamic


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
    temp_dir = project_root / "temp_internal_docs"
    temp_dir.mkdir(exist_ok=True)
    try:
        uploaded_count = 0
        processed_files = []
        for uploaded_file in uploaded_files:
            file_extension = uploaded_file.name.lower().split('.')[-1]
            if file_extension in ['txt', 'md', 'markdown']:
                file_path = temp_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                uploaded_count += 1
                processed_files.append(uploaded_file.name)
            elif file_extension == 'zip':
                try:
                    with zipfile.ZipFile(io.BytesIO(uploaded_file.getbuffer()), 'r') as zip_ref:
                        for file_info in zip_ref.filelist:
                            if file_info.filename.lower().endswith(('.txt', '.md', '.markdown')):
                                zip_ref.extract(file_info, temp_dir)
                                uploaded_count += 1
                                processed_files.append(f"{uploaded_file.name}/{file_info.filename}")
                except Exception as e:
                    st.warning(f"Error processing ZIP file {uploaded_file.name}: {e}")
            else:
                st.warning(f"Skipped {uploaded_file.name} - Only .txt, .md, .markdown and .zip files are supported")
        if uploaded_count > 0:
            indexed_count = index_internal_data_with_source(
                str(temp_dir),
                source_name or "uploaded_docs"
            )
            shutil.rmtree(temp_dir, ignore_errors=True)
            return indexed_count, processed_files
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return 0, []
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        st.error(f"Error during indexing: {e}")
        return 0, []


def submit_completed_rfp(current_file, completed_df, submitter_name, source=None):
    """Submit completed RFP: save to Excel, index in vector DB (embeddings generated inside upsert_rfp), and manage file movement"""
    try:
        source_path = project_root / "data" / "new_RFPs" / current_file
        outputs_folder = project_root / "outputs"
        past_rfps_folder = project_root / "data" / "past_RFPs_pdf"
        completed_rfps_folder = project_root / "data" / "completed_RFPs_excel"

        outputs_folder.mkdir(parents=True, exist_ok=True)
        past_rfps_folder.mkdir(parents=True, exist_ok=True)
        completed_rfps_folder.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_parts = current_file.rsplit('.', 1)
        excel_filename = f"completed_{name_parts[0]}_{timestamp}.xlsx"
        excel_filepath = outputs_folder / excel_filename

        metadata_df = completed_df.copy()
        metadata_df['Submitter Name'] = submitter_name

        metadata_df.to_excel(excel_filepath, index=False)

        if upsert_rfp is None:
            st.error("Vector indexing not available - upsert_rfp function not loaded. Vérifiez l'import dans qdrant/indexer.py.")
            return False, "Vector indexing not available - upsert_rfp function not loaded.", excel_filename, 0
        try:
            indexed_count = upsert_rfp(
                rfp_file_path=str(excel_filepath),
                submitter_name=submitter_name,
                source=source
            )
            print(f"✅ Successfully indexed {indexed_count} Q&A pairs (embeddings generated internally)")
            print(f"📝 Metadata: Submitter={submitter_name}")
        except Exception as e:
            st.warning(f"RFP saved but vector indexing failed: {e}")
            print(f"❌ Vector indexing error: {e}")
            indexed_count = 0

        success_message = ""
        if indexed_count > 0:
            if source_path.exists():
                dest_pdf_path = past_rfps_folder / current_file
                shutil.move(str(source_path), str(dest_pdf_path))
                print(f"✅ Moved PDF: {current_file} -> past_RFPs_pdf")
                success_message += f"PDF moved to past_RFPs_pdf. "

            completed_excel_path = completed_rfps_folder / excel_filename
            shutil.copy2(str(excel_filepath), str(completed_excel_path))
            print(f"✅ Saved Excel: {excel_filename} -> completed_RFPs_excel")
            success_message += f"Excel saved to completed_RFPs_excel. "

            return True, f"Successfully submitted! {success_message}Indexed {indexed_count} Q&A pairs.", excel_filename, indexed_count
        else:
            return False, "No questions were indexed. Files not moved.", excel_filename, 0

    except Exception as e:
        return False, f"Error during submission: {str(e)}", None, 0


def parse_selected_rfp(selected_file):
    """Parse the selected RFP file and return questions"""
    rfp_path = project_root / "data" / "new_RFPs" / selected_file
    try:
        rfp_parser_agent = RFPParserAgent()
        questions_list, excel_path = rfp_parser_agent.parse_rfp_file(str(rfp_path))
        df_data = []
        for question in questions_list:
            df_data.append({
                'Question': str(question),
                'Answer': '',
                'Comments': ''
            })
        df = pd.DataFrame(df_data)
        df = df.astype({'Question': 'string', 'Answer': 'string', 'Comments': 'string'})
        return df
    except Exception as e:
        st.error(f"Error parsing RFP: {e}")
        return None


def show_pdf_viewer(pdf_file_path):
    """Display PDF file in the Streamlit interface"""
    try:
        with open(pdf_file_path, "rb") as f:
            pdf_bytes = f.read()
        pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
        pdf_display = f"""
        <iframe src="data:application/pdf;base64,{pdf_base64}" 
                width="100%" 
                height="600px" 
                style="border: 1px solid #ccc; border-radius: 5px;">
        </iframe>
        """
        st.markdown(pdf_display, unsafe_allow_html=True)
        file_size = len(pdf_bytes) / (1024 * 1024)
        st.info(f"**File**: {Path(pdf_file_path).name} | **Size**: {file_size:.2f} MB")
        return True
    except Exception as e:
        st.error(f"Could not display PDF: {e}")
        st.info("💡 The file might be corrupted or not a valid PDF format")
        return False


def pre_complete_rfp(current_df, mode="dev"):
    """Pre-complete RFP answers using ReAct-based AI retriever"""
    if get_qdrant_client is None:
        st.error("ReAct retriever components not available")
        return current_df
    try:
        session_key = f'completion_agent_{mode}'
        if session_key not in st.session_state or st.session_state[session_key] is None:
            with st.spinner(f"🔧 Setting up ReAct AI system in {mode.upper()} mode..."):
                setup_collections_dynamic()
                if mode == "prod" and not settings.OPENAI_API_KEY:
                    st.error("Production mode requires OpenAI API key. Set OPENAI_API_KEY in .env file.")
                    return current_df
                completion_agent = RFPCompletionAgent(mode="prod")
                st.session_state[session_key] = completion_agent
        else:
            completion_agent = st.session_state[session_key]

        completed_df = current_df.copy()
        total_questions = len(completed_df)
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, row in completed_df.iterrows():
            question = row['Question']
            if not question or len(question.strip()) < 10:
                continue
            progress = (idx + 1) / total_questions
            progress_bar.progress(progress)
            status_text.text(f"ReAct {mode.upper()} processing question {idx + 1}/{total_questions}: {question[:50]}...")
            try:
                result = completion_agent.answer_question(question)
                binary_answer = result.get('answer', 'No')  # Yes/No
                detailed_comments = result.get('comments', 'No detailed explanation provided')
                completed_df.at[idx, 'Answer'] = binary_answer
                mode_info = f"ReAct AI ({mode.upper()})"
                if row['Comments'] and row['Comments'].strip():
                    completed_df.at[idx, 'Comments'] = f"{row['Comments'].strip()} | {mode_info}: {detailed_comments}"
                else:
                    completed_df.at[idx, 'Comments'] = f"{mode_info}: {detailed_comments}"
            except Exception as e:
                st.warning(f"Could not process question {idx + 1} with ReAct: {e}")
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
    .main > div { padding-top: 1rem; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; background-color: #f8f9fa; padding: 6px; border-radius: 10px; margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab"] { padding: 12px 24px; font-weight: 500; border-radius: 6px; color: #495057; border: none; }
    .stTabs [data-baseweb="tab"]:hover { background-color: #e9ecef; color: #495057; }
    .stTabs [aria-selected="true"] { background-color: #007bff !important; color: white !important; }
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 8px; color: white; text-align: center; margin-bottom: 1rem; }
    .section-header { border-bottom: 2px solid #e9ecef; padding-bottom: 0.5rem; margin-bottom: 1.5rem; font-weight: 600; color: #2c3e50; }
    .stButton > button { border-radius: 6px; font-weight: 500; transition: all 0.3s ease; }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
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
    - **Rich Metadata**: Answers, comments, submitter info, and completion date stored
    - **Future AI Enhancement**: Completed RFPs improve future pre-completion accuracy
    - **Knowledge Accumulation**: Building organizational RFP expertise over time
    - **Accountability**: Full traceability of who validated each Q&A and who submitted the RFP
    - **Age Management**: Automatic cleanup of old RFPs (configurable, default 20 RFPs)
    ### Key Features Overview:
    """)
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
        - **Accountability**: Track submitter info
        - **Vector Database**: Smart indexing for future improvements
        """)
    st.markdown("---")
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
    with st.expander("Tips & Best Practices", expanded=False):
        st.markdown("""
        ### For Best Results:
        1. **Upload Quality PDFs**: Clear, text-based PDFs work best
        2. **Build Knowledge Base**: Upload internal documentation before processing
        3. **Review AI Suggestions**: Always validate AI-generated answers
        4. **Regular Cleanup**: Monitor RFP statistics and cleanup old data periodically
        ### Troubleshooting:
        - **No Questions Found**: Check PDF quality and format
        - **AI Not Working**: Verify Ollama is running (dev mode) or API keys (prod mode)
        - **Slow Performance**: Consider reducing batch size or upgrading hardware
        - **Database Issues**: Use the cleanup tools in sidebar statistics section
        """)


def show_pdf_bytes_viewer(pdf_bytes: bytes, filename: str = "uploaded.pdf") -> bool:
    """Display PDF bytes in an iframe (no disk write to permanent folders)."""
    try:
        pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
        pdf_display = f"""
        <iframe src="data:application/pdf;base64,{pdf_base64}"
                width="100%" height="600px"
                style="border: 1px solid #ccc; border-radius: 5px;">
        </iframe>
        """
        st.markdown(pdf_display, unsafe_allow_html=True)
        file_size = len(pdf_bytes) / (1024 * 1024)
        st.info(f"**File**: {filename} | **Size**: {file_size:.2f} MB")
        return True
    except Exception as e:
        st.error(f"Could not display PDF from memory: {e}")
        return False


def parse_rfp_from_path(rfp_path_str: str):
    """Parse an RFP from an absolute file path and return a Q/A/Comments DataFrame (same schema as parse_selected_rfp)."""
    try:
        rfp_parser_agent = RFPParserAgent()
        questions_list, _excel_path = rfp_parser_agent.parse_rfp_file(str(rfp_path_str))
        df = pd.DataFrame(
            [{"Question": str(q), "Answer": "", "Comments": ""} for q in questions_list],
            dtype="string"
        )
        df = df.astype({"Question": "string", "Answer": "string", "Comments": "string"})
        return df
    except Exception as e:
        st.error(f"Error parsing RFP (path): {e}")
        return None


def show_rfp_manager():
    """Main RFP Manager interface:
    - Sidebar: batch drag & drop for PDFs (kept as-is)
    - Main (top): single-file upload (no save to new_RFPs) OR pick from folder
    """

    # Header with RFP counter
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("RFP Question Manager")

    ss = st.session_state
    ss.setdefault("current_file_source", None)
    ss.setdefault("uploaded_pdf_bytes", None)
    ss.setdefault("uploaded_pdf_name", None)
    ss.setdefault("temp_pdf_path", None)
    ss.setdefault("folder_selected_file", None)
    # NEW: nonce to force remount/clear of the single-file uploader
    ss.setdefault("single_uploader_nonce", 0)
    # (Optional) nonce for sidebar batch uploader if you ever want to reset it too
    ss.setdefault("sidebar_uploader_nonce", 0)

    st.markdown("---")

    # ============ SIDEBAR: Batch upload (unchanged) ============
    st.sidebar.title("File Management")
    st.sidebar.subheader("Upload New RFPs by batch")

    uploaded_files_sidebar = st.sidebar.file_uploader(
        "Drag & drop multiple PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload PDF files to RFPs folder",
        key=f"rfp_uploader_sidebar_{ss.sidebar_uploader_nonce}"
    )
    if uploaded_files_sidebar:
        uploaded_count = handle_file_upload(uploaded_files_sidebar)
        if uploaded_count > 0:
            st.sidebar.success(f"Uploaded {uploaded_count} PDF file(s)")
            st.rerun()
        else:
            st.sidebar.warning("No PDF files were uploaded.")

    st.sidebar.markdown("---")

    # ====== Sidebar: Knowledge Base Management (as you had) ======
    st.sidebar.header("Knowledge Base Management")
    source_name = st.sidebar.text_input(
        "Source Name",
        value="",
        help="Name to identify this batch of documents",
        key="source_name_files"
    )
    internal_docs = st.sidebar.file_uploader(
        "Zip containing html and folders",
        type=['zip'],
        accept_multiple_files=True,
        help="Upload ZIP archives",
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
                    st.sidebar.warning("No documents were indexed")

    # ============ MAIN: Single upload OR pick existing ============
    st.subheader("Start here")
    st.markdown("""
<div style="
    background-color: #f0f8ff;
    border: 1px solid #cce7ff;
    border-radius: 10px;
    padding: 20px;
    margin: 15px 0;
    font-size: 1.1rem;
    line-height: 1.6;
">
<b>How to prompt</b><br><br>
- On the <b>left</b>: drag & drop or browse to upload <b>a single PDF RFP to select it.</b>.  
<br>
- On the <b>right</b>: choose an <b>existing RFP</b> already stored in the folder.  
<br>
Once you make a choice, scroll down to <b>preview the PDF</b> and <b>process the RFP</b>.
</div>
""", unsafe_allow_html=True)

    # Reset button to clear any current selection
    if st.button("Reset RFP Selection", help="Clear the currently selected or uploaded RFP"):
        # Clear uploaded file state
        ss.uploaded_pdf_bytes = None
        ss.uploaded_pdf_name = None

        # Clear folder selection
        ss.folder_selected_file = None

        # Reset active source
        ss.current_file_source = None

        # Remove temp file if it exists
        if ss.temp_pdf_path and os.path.exists(ss.temp_pdf_path):
            try:
                os.remove(ss.temp_pdf_path)
            except Exception:
                pass
        ss.temp_pdf_path = None

        # Clear parsed/editor state
        for k in ['rfp_data', 'current_file', 'rfp_question_vectors']:
            if k in ss:
                del ss[k]

        # NEW: force-remount the single-file uploader (and sidebar if desired)
        ss.single_uploader_nonce += 1
        # If you also wanted to clear the sidebar batch uploader, uncomment next line:
        # ss.sidebar_uploader_nonce += 1

        # Refresh UI
        st.rerun()

    left, right = st.columns(2)

    # Load available RFP files from folder
    rfp_files = load_rfp_files()

    # Session keys for selection state (redundant safety)
    ss.setdefault("current_file_source", None)
    ss.setdefault("uploaded_pdf_bytes", None)
    ss.setdefault("uploaded_pdf_name", None)
    ss.setdefault("temp_pdf_path", None)
    ss.setdefault("folder_selected_file", None)

    # LEFT: Single-file upload (no save to new_RFPs) — uses a dynamic key
    with left:
        single_pdf = st.file_uploader(
            "Upload one RFP (PDF from your computer)",
            type=["pdf"],
            accept_multiple_files=False,
            help="This file will NOT be saved to data/new_RFPs/. It will be used directly.",
            key=f"rfp_single_uploader_main_{ss.single_uploader_nonce}"
        )

        if single_pdf is not None:
            ss.current_file_source = "uploaded"
            ss.uploaded_pdf_bytes = single_pdf.getbuffer().tobytes()
            ss.uploaded_pdf_name = single_pdf.name

            # Create/refresh a temporary file path for parsers that need a path
            try:
                if ss.temp_pdf_path and os.path.exists(ss.temp_pdf_path):
                    try:
                        os.remove(ss.temp_pdf_path)
                    except Exception:
                        pass
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(ss.uploaded_pdf_bytes)
                    ss.temp_pdf_path = tmp.name
            except Exception as e:
                st.error(f"Failed to create a temporary PDF file: {e}")
                ss.temp_pdf_path = None

            # Clear folder selection to avoid ambiguity
            ss.folder_selected_file = None

    # RIGHT: Folder picker
    with right:
        if not rfp_files:
            st.warning("No RFP files found in data/new_RFPs/.")
        else:
            options = ["— Select —"] + rfp_files
            current_index = 0
            if ss.folder_selected_file in rfp_files:
                current_index = options.index(ss.folder_selected_file)
            picked = st.selectbox(
                "Pick an existing RFP from the RFPs Folder.",
                options=options,
                index=current_index,
                help="Choose a PDF file already present in the folder.",
                key="main_rfp_selector"
            )
            if picked != "— Select —":
                if ss.current_file_source != "uploaded":
                    ss.current_file_source = "folder"
                    ss.folder_selected_file = picked
                    ss.uploaded_pdf_bytes = None
                    ss.uploaded_pdf_name = None
                    if ss.temp_pdf_path and os.path.exists(ss.temp_pdf_path):
                        try:
                            os.remove(ss.temp_pdf_path)
                        except Exception:
                            pass
                        ss.temp_pdf_path = None

    st.markdown("---")

    # ============ PREVIEW & EXTRACT ============
    active_source = ss.current_file_source

    if active_source == "uploaded" and ss.uploaded_pdf_bytes:
        st.subheader("PDF Preview (Uploaded)")
        show_pdf_bytes_viewer(ss.uploaded_pdf_bytes, ss.uploaded_pdf_name or "uploaded.pdf")

        st.markdown("---")
        st.subheader("Extract Questions")

        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            if st.button("Extract questions from uploaded PDF", type="primary", use_container_width=True):
                if not ss.temp_pdf_path:
                    st.error("Temporary file not available for parsing.")
                else:
                    with st.spinner("Parsing uploaded RFP..."):
                        df = parse_rfp_from_path(ss.temp_pdf_path)
                        if df is not None:
                            st.session_state.rfp_data = df
                            st.session_state.current_file = f"(Uploaded) {ss.uploaded_pdf_name or 'RFP.pdf'}"
                            st.success(f"Successfully parsed {len(df)} questions from uploaded PDF")
                            st.rerun()
                        else:
                            st.error("Failed to parse the uploaded RFP.")

    elif active_source == "folder" and ss.folder_selected_file:
        st.subheader("PDF Preview")
        rfp_path = project_root / "data" / "new_RFPs" / ss.folder_selected_file
        with st.spinner("Loading PDF preview..."):
            show_pdf_viewer(str(rfp_path))

        st.markdown("---")
        st.subheader("Extract Questions")

        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            if st.button("Extract questions from RFP", type="primary", use_container_width=True):
                with st.spinner("Parsing RFP file..."):
                    df = parse_selected_rfp(ss.folder_selected_file)
                    if df is not None:
                        st.session_state.rfp_data = df
                        st.session_state.current_file = ss.folder_selected_file
                        st.success(f"Successfully parsed {len(df)} questions from {ss.folder_selected_file}")
                        st.rerun()
                    else:
                        st.error("Failed to parse the RFP file. Please check the file format.")

    else:
        st.info("Upload a single PDF on the left cell OR pick an existing one on the right cell to continue.")

    st.markdown("---")

    # ============ EDITING INTERFACE ============
    if 'rfp_data' in st.session_state:
        st.header(f"Current RFP: {st.session_state.current_file}")

        st.markdown("### Questions, Answers & Comments")
        edited_df = st.data_editor(
            st.session_state.rfp_data,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "Question": st.column_config.TextColumn("Question", width="large", help="RFP questions extracted from the document"),
                "Answer": st.column_config.SelectboxColumn("Answer", width="small", options=["", "Yes", "No"], help="Select Yes or No for each question"),
                "Comments": st.column_config.TextColumn("Comments", width="large", help="Add comments or explanations for each answer"),
            },
            hide_index=True,
            key="rfp_editor"
        )
        st.session_state.rfp_data = edited_df

        st.markdown("### Actions")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Pre-complete", type="primary", use_container_width=True,
                         help="Auto-fill answers using Reasoning and Action"):
                if get_qdrant_client is None:
                    st.error("ReAct retriever components not available")
                else:
                    if not settings.AZURE_OPENAI_API_KEY:
                        st.error("Azure OpenAI API key required. Please set AZURE_OPENAI_API_KEY in .env file.")
                    else:
                        try:
                            setup_collections_dynamic()
                            completion_agent = RFPCompletionAgent()
                            processed_df = edited_df.copy()

                            questions_to_process = []
                            for index, row in processed_df.iterrows():
                                question = row['Question']
                                if (pd.notna(question) and str(question).strip() and
                                    (pd.isna(row['Answer']) or not str(row['Answer']).strip() or
                                     str(row['Answer']).strip() in ['', 'nan'])):
                                    questions_to_process.append(index)

                            if len(questions_to_process) == 0:
                                st.info("All questions already have answers. Clear the Answer column if you want to reprocess.")
                            else:
                                progress_placeholder = st.empty()
                                processed_count = 0
                                question_vectors = []

                                for index in questions_to_process:
                                    q = processed_df.at[index, 'Question']
                                    processed_count += 1
                                    progress_placeholder.info(f"Processing: {processed_count}/{len(questions_to_process)} questions")

                                    try:
                                        result = completion_agent.answer_question(str(q))
                                        answer = result.get('answer', 'No')
                                        comments = result.get('comments', 'Please review this section')
                                        processed_df.at[index, 'Answer'] = answer
                                        processed_df.at[index, 'Comments'] = comments
                                        question_vectors.append([])
                                    except Exception:
                                        processed_df.at[index, 'Answer'] = "No"
                                        processed_df.at[index, 'Comments'] = "Please review this section"
                                        question_vectors.append([])

                                st.session_state.rfp_question_vectors = question_vectors
                                st.session_state.rfp_data = processed_df
                                progress_placeholder.success(
                                    f"Completed: {processed_count} questions processed with ReAct AI! ({len(question_vectors)} vectors collected)"
                                )
                                st.rerun()
                        except Exception as e:
                            st.error(f"AI processing failed: {e}")

        with col2:
            try:
                import io
                buffer = io.BytesIO()
                edited_df.to_excel(buffer, index=False, engine='openpyxl')
                buffer.seek(0)
                base_name = (st.session_state.current_file or "rfp").replace(".pdf", "")
                filename_download = f"rfp_edited_{base_name}.xlsx"
                st.download_button(
                    label="Download Excel",
                    data=buffer.getvalue(),
                    file_name=filename_download,
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    use_container_width=True,
                    help="Download the Excel file to your computer"
                )
            except Exception as e:
                st.error(f"Download preparation failed: {e}")

        with col3:
            st.markdown("**Submit Information**")
            submitter_name = st.text_input(
                "Submitter Name",
                placeholder="Enter submitter name",
                help="Name of the person submitting this RFP",
                key="submitter_name_input"
            )
            if st.button("Submit", type="primary", use_container_width=True,
                         help="Complete RFP: Save and Archive"):
                if not submitter_name or not submitter_name.strip():
                    st.error("Please enter the Submitter Name")
                else:
                    with st.spinner("Submitting RFP..."):
                        success, result, excel_file, indexed_count = submit_completed_rfp(
                            st.session_state.current_file if isinstance(st.session_state.current_file, str) else "",
                            edited_df,
                            submitter_name=submitter_name.strip(),
                            source=None
                        )

                        if success:
                            st.success(f"""
**RFP Successfully Submitted!**

**Submitted by**: {submitter_name.strip()}
**Archived as**: {result}
**Excel saved**: {excel_file}
**Vector DB indexed**: {indexed_count} Q&A pairs

The RFP knowledge is now increased for future processing!
""")
                            for k in ['rfp_data', 'current_file', 'rfp_question_vectors',
                                      'uploaded_pdf_bytes', 'uploaded_pdf_name', 'folder_selected_file',
                                      'current_file_source']:
                                if k in st.session_state:
                                    del st.session_state[k]
                            if st.session_state.get('temp_pdf_path') and os.path.exists(st.session_state['temp_pdf_path']):
                                try:
                                    os.remove(st.session_state['temp_pdf_path'])
                                except Exception:
                                    pass
                                st.session_state['temp_pdf_path'] = None

                            # Ensure uploader is cleared after submit as well
                            st.session_state['single_uploader_nonce'] += 1
                            st.rerun()
                        else:
                            st.error(f"Failed to submit RFP: {result}")

        st.markdown("---")
        col4, col5, col6 = st.columns(3)
        with col4:
            if st.button("Reset", help="Clear Answer and Comments columns"):
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
            pass

    if 'rfp_data' not in st.session_state:
        st.info("Extract questions from the selected RFP to continue")


if __name__ == "__main__":
    main()
