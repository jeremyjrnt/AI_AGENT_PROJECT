#!/bin/bash
# Launcher script for RFP Manager UI
echo "🚀 Starting RFP Manager..."
cd "$(dirname "$0")/.."
source venv/bin/activate
streamlit run ui/rfp_manager.py --server.port 8501
