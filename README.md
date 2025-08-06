# RFPilot

## Overview
This project is an AI-Agent designed for completing RFPs (Request for Proposals) using a Retrieval-Augmented Generation (RAG) pipeline.

## Requirements
- Python 3.8+
- Docker
- Qdrant

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd RFPilot
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up Qdrant using Docker with a shared folder for persistent storage:
   ```bash
   mkdir -p ./qdrant_data
   docker run -p 6333:6333 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant
   ```

## Usage
- To run the pipeline server, execute:
   ```bash
   uvicorn RFPilot.scripts.server:app --host 0.0.0.0 --port 8000
   ```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.