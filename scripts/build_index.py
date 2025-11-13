#!/usr/bin/env python3
"""
build_index.py - Build Pinecone vector index from processed documents

This script loads cleaned documents and populates the Pinecone index.
Run this after ingest_and_filter.py to build the vector store.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from scripts.ingest_and_filter import to_langchain_documents
from src.rag_agent_library import (
    get_pinecone_vectorstore,
    EMBEDDING_MODEL_NAME
)
from langchain_together import TogetherEmbeddings
from dotenv import load_dotenv

load_dotenv()

def main():
    """Build the Pinecone index from processed documents."""
    print("Building Pinecone index...")

    # Load processed data
    df = pd.read_csv("./data/processed/clean_questions.csv")
    docs = to_langchain_documents(df)
    print(f"Loaded {len(docs)} documents from processed data.")

    # Initialize embeddings
    embeddings = TogetherEmbeddings(model=EMBEDDING_MODEL_NAME)

    # Get/create vector store and populate it
    print("Connecting to Pinecone and populating index...")
    vectorstore = get_pinecone_vectorstore(embeddings, force_reseed=True)

    if vectorstore:
        print(f"✅ Pinecone index populated with {len(docs)} documents.")
    else:
        print("❌ Failed to build Pinecone index.")

if __name__ == "__main__":
    main()
