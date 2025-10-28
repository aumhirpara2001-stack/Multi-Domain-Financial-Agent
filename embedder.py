# embedder.py
"""
Build a FAISS vector store from a pandas DataFrame.

Flow (mathematical / pipeline logic):
1. Text documents D = {d1, d2, ...}
2. Splitter S divides each di into chunks c_ij with chunk_size and chunk_overlap
   -> Chunks C = {c_11, c_12, ..., c_nm}
3. Embedding function E: text -> R^k (embedding vector)
   For each chunk c in C compute v = E(c)
4. Vector store V stores pairs (v, metadata)
5. Retriever R queries V for nearest neighbor vectors to a query q by computing
   E(q) and returning top-k chunks.
6. LLM uses retrieved context + query to produce answer.

This file implements steps 2-4 and returns a LangChain FAISS VectorStore object.
"""

import os
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

# LangChain imports (0.3.x API surface)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

# Embedding imports (OpenAI and optional Together)
try:
    from langchain.embeddings.openai import OpenAIEmbeddings
except Exception:
    OpenAIEmbeddings = None

# Attempt to import Together embeddings if available in environment
# Note: package name langchain-together provides Together integrations; exact import paths vary.
# We'll attempt a common path, otherwise we raise a helpful error if chosen.
try:
    # Some distributions expose TogetherEmbeddings under langchain_together.embeddings
    from langchain_together.embeddings import TogetherEmbeddings  # type: ignore
except Exception:
    try:
        # alternate guess
        from langchain_together import TogetherEmbeddings  # type: ignore
    except Exception:
        TogetherEmbeddings = None

load_dotenv()

EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "openai").lower()


def _get_embedding_instance():
    """
    Instantiate and return the embeddings object according to EMBEDDING_PROVIDER.
    If OpenAI requested but OpenAIEmbeddings isn't importable, raise an error with guidance.
    """
    if EMBEDDING_PROVIDER == "openai":
        if OpenAIEmbeddings is None:
            raise ImportError("OpenAIEmbeddings not available. Ensure langchain-openai is installed.")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY missing in environment.")
        # OpenAIEmbeddings accepts api_key via env var; optionally pass credentials here.
        return OpenAIEmbeddings()
    elif EMBEDDING_PROVIDER in ("together", "togetherai"):
        if TogetherEmbeddings is None:
            raise ImportError(
                "TogetherEmbeddings not available. Install langchain-together and "
                "ensure the package exposes TogetherEmbeddings in this environment."
            )
        api_key = os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise EnvironmentError("TOGETHER_API_KEY missing in environment.")
        # instantiate TogetherEmbeddings according to provider's constructor
        return TogetherEmbeddings(api_key=api_key)
    else:
        raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}")


def build_vector_store(df: pd.DataFrame, text_column: str = "text", chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Build a FAISS vector store from a pandas DataFrame column.

    Args:
        df: pandas DataFrame containing source documents (must contain `text_column`).
        text_column: name of the column with raw text.
        chunk_size: maximum characters per chunk.
        chunk_overlap: overlap between chunks in characters.

    Returns:
        langchain.vectorstores.FAISS object (in-memory).
    """

    # Validate DataFrame
    if text_column not in df.columns:
        raise ValueError(f"DataFrame must contain a '{text_column}' column.")

    # 1) Split documents into chunks
    # RecursiveCharacterTextSplitter splits based on characters while trying to respect sentence boundaries.
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    docs = []
    for idx, row in df.iterrows():
        raw_text = str(row[text_column])
        # Split into chunks; returns list[str]
        chunks = splitter.split_text(raw_text)
        for i, chunk in enumerate(chunks):
            # Attach metadata so we can trace provenance
            metadata = {
                "source_index": int(idx),
                "chunk_index": i,
            }
            docs.append(Document(page_content=chunk, metadata=metadata))

    # 2) Get embedding function based on provider
    embeddings = _get_embedding_instance()

    # 3) Build FAISS index
    # FAISS.from_documents handles embedding computation and index construction
    vecstore = FAISS.from_documents(docs, embeddings)

    return vecstore