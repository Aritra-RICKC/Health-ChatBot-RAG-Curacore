# memory_llm.py

# step1: Load raw PDFs (one by one)
# step2: Create chunks from the loaded PDF
# step3: Create vector embeddings for the chunks
# step4: Add embeddings to a FAISS vector store incrementally and save

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pypdf

# This line increases the decompression limit for pypdf to handle large, complex PDFs.
pypdf.constants.DECOMPRESSION_LIMIT = 2 * 1024**3 

# --- Configuration ---
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

# --- Helper Functions ---

def create_chunks(documents):
    """Splits loaded documents into smaller chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                 chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

def get_embedding_model():
    """Initializes and returns the embedding model from HuggingFace."""
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

# --- Main Processing Logic ---

def main():
    """
    Main function to process PDFs incrementally and build the vector store.
    """
    print("🚀 Starting the document processing pipeline...")

    # Initialize the embedding model once
    print("Initializing embedding model...")
    embedding_model = get_embedding_model()
    print("Embedding model loaded successfully.")

    # Get a list of all PDF files in the specified directory
    try:
        pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
        if not pdf_files:
            print(f" No PDF files found in the '{DATA_PATH}' directory. Exiting.")
            return
    except FileNotFoundError:
        print(f" Error: The directory '{DATA_PATH}' was not found. Please create it and add your PDFs. Exiting.")
        return

    print(f"Found {len(pdf_files)} PDF files to process.")

    db = None

    # Process each PDF file one by one to conserve memory
    for i, filename in enumerate(pdf_files):
        file_path = os.path.join(DATA_PATH, filename)
        print(f"\n--- Processing file {i+1}/{len(pdf_files)}: {filename} ---")

        try:
            # Step 1: Load a single PDF
            print(f"Loading PDF: {file_path}...")
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            print(f"Loaded {len(documents)} pages from the document.")

            # Step 2: Create chunks for the current document
            print("Creating text chunks...")
            text_chunks = create_chunks(documents)
            print(f"Created {len(text_chunks)} chunks.")

            if not text_chunks:
                print("No text chunks were generated, skipping this file.")
                continue

            # Step 3 & 4: Embed chunks and add to the vector store
            print("Embedding chunks and updating the vector store...")
            if db is None:
                # For the first document, create a new FAISS index
                db = FAISS.from_documents(text_chunks, embedding_model)
                print("Created a new FAISS index.")
            else:
                # For subsequent documents, add their vectors to the existing index
                db.add_documents(text_chunks)
                print("Added new chunks to the existing FAISS index.")

        except Exception as e:
            print(f" An error occurred while processing {filename}: {e}")
            print("Skipping this file and moving to the next one.")
            continue
    
    # After processing all files, save the final vector store
    if db:
        print("\n--- Finalizing ---")
        print(f"Saving the final FAISS index to: {DB_FAISS_PATH}")
        db.save_local(DB_FAISS_PATH)
        print("Vector store saved successfully.")
    else:
        print("\nNo documents were successfully processed. No vector store was saved.")

if __name__ == "__main__":
    main()