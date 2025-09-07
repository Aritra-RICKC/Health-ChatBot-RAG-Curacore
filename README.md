# Curacore - Medical Chatbot with RAG

A conversational AI assistant for medical information retrieval using Retrieval-Augmented Generation (RAG) with FAISS vector database.

## Features

- Medical knowledge retrieval from PDF documents
- Conversational AI interface
- Vector similarity search using FAISS
- Streamlit-based web interface

## Project Structure

```
├── .env                - Environment variables
├── chatbot.py          - Main chatbot application
├── connect_memory.py   - Memory connection utilities
├── memory_llm.py       - LLM memory management
├── data/               - Medical PDF documents
├── vectorstore/        - FAISS vector database
└── requirements.txt    - Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Aritra-RICKC/Health-ChatBot-RAG-Curacore.git
cd Healthbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env` file

## Usage

Run the Streamlit application:
```bash
streamlit run chatbot.py
```

## Data Sources

The bot currently uses these medical references:
- The GALE Encyclopedia of Medicine (2nd Edition)
- Parks' Preventive and Social Medicine (23rd Edition)

## Configuration

Edit `.streamlit/config.toml` to customize the web interface appearance.

## License

This project is licensed under the terms of the MIT license - see the [LICENSE](LICENSE) file for details.
