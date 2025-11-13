
#rag_agent_library.py

"""
rag_agent_library.py - ANNOTATED VERSION

This file is our "RAG Brain" as a library.
It does NOT run on its own. It is *imported* by other scripts
(like `chat_cli.py`) that want to use its functions.

It contains two main "public" functions:
1.  get_pinecone_vectorstore(...): Connects to/builds the agent's "memory."
2.  create_rag_pipeline(...): Creates the agent's "brain."
"""

# --- Section 1: Imports, Configuration & Environment Setup ---
# =============================================================

import os
import pandas as pd
import time
from typing import List, Tuple, Dict, Any, Optional

# --- NEW: Load .env file ---
# This line automatically finds your .env file and loads all
# the keys into os.environ.
from dotenv import load_dotenv
load_dotenv()
# --- END NEW ---

# --- Core LangChain Imports (v0.2.x+ "core" and "community") ---
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseChatModel

# We need a text splitter to break up long documents
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LLMs and Embeddings
from langchain_together import TogetherEmbeddings, ChatTogether

# Vector Stores
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# LCEL Components
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnablePassthrough
)
# --- END Core LangChain Imports ---

# --- Foundation: Call Generator for Evals to run ---
# ========================================  
import requests

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")  # or hardcode for dev

def call_rag_generator(query: str, chunks: list[str]) -> dict:
    context = "\n\n".join(chunks)
    prompt = f"""You are a financial analyst. Use the context below to answer the question.

Context:
{context}

Question:
{query}

Answer:"""

    response = requests.post(
        "https://api.together.xyz/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "togethercomputer/llama-2-70b-chat",  # or your preferred model
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 512
        }
    )

    result = response.json()
    return {"text": result["choices"][0]["message"]["content"].strip()}

# --- 1.1: Prompt Constants ---
RAG_SYSTEM_PROMPT = """
You are an LLM Agentic Retrieval-Augmented Generation (RAG) assistant.
Your role is to answer financial and strategic questions by following this order of operations:

1. **Assess Context Need**
   - Decide if external retrieval is required.
   • If the question can be answered directly (general knowledge, math, definitions), answer without retrieval.
   • If the question requires specific financial or strategic context, query the vector store.

2. **Clarify the Question**
   - Reformulate if needed, or ask a follow-up if context is missing or ambiguous.

3. **Retrieve Context (if needed)**
   - Query the vector store and gather the most relevant documents.

4. 4. **Ground the Answer with Mathematical Rigor**
   - Synthesize retrieved context into a concise, accurate response.
   - Ensure reasoning is mathematically rigorous and logically defensible.
   - Cite sources when available.
   - When numeric reasoning is required, present the derivation clearly and concisely using LaTeX formatting. Avoid unnecessary verbosity or over-explaining basic steps.


5. **Handle Uncertainty**
   - If retrieval is weak or conflicting, state uncertainty and ask for clarification instead of guessing.

6. **Deliver the Response**
   - Present the Principal Finding first.
   - Support with structured reasoning, bullet points, and optional LaTeX for equations.

Tone: Neutral, rigorous, professional — as if writing for a finance exam grader.
"""

CONTEXTUALIZE_SYSTEM_PROMPT = """
You are a contextualizer assistant. Your job is to reformulate ambiguous user questions into standalone questions that do not rely on chat history.

Only reformulate if the question contains vague references like 'this', 'that', 'above', or 'earlier'. Otherwise, return the question unchanged.

Do NOT answer the question. Just return the reformulated version.
"""

# --- 1.2: Configuration ---
CSV_FILE_PATH = "data/raw/all_questions_tagged.csv"

EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL_NAME",
    "BAAI/bge-base-en-v1.5"
)
EMBEDDING_DIM_MAP = {
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-large-en-v1.5": 1024
}
EMBEDDING_DIM = EMBEDDING_DIM_MAP.get(EMBEDDING_MODEL_NAME, 768)

LLM_MODEL_NAME = os.environ.get(
    "LLM_MODEL_NAME",
    "meta-llama/Llama-3-70b-chat-hf"
)
PINECONE_INDEX_NAME = os.environ.get(
    "PINECONE_INDEX_NAME",
    "agentic-rag-index" # Renamed to something more permanent
)


# --- Section 2: Data Ingestion & Indexing (Retrieval) ---
# =========================================================

def load_csv_docs(csv_path: str = CSV_FILE_PATH) -> List[Document]:
    """
    Loads the CSV and turns each row into a LangChain 'Document'.
    """
    print(f"Loading documents from: {csv_path}")
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return []
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return []

    documents = []
    required_cols = ['id', 'question', 'answer', 'context',
                       'source_dataset', 'primary_tag', 'secondary_tags']

    if not all(col in df.columns for col in required_cols):
        print("Error: CSV is missing one or more required columns.")
        return []

    for _, row in df.iterrows():
        content = (
            f"Question: {row['question']}\n\n"
            f"Answer: {row['answer']}\n\n"
            f"Context: {row['context']}"
        )
        metadata = {
            'id': str(row.get('id', 'N/A')),
            'question': str(row.get('question', 'N/A')),
            'answer': str(row.get('answer', 'N/A')),
            'context': str(row.get('context', 'N/A')),
            'source_dataset': str(row.get('source_dataset', 'N/A')),
            'primary_tag': str(row.get('primary_tag', 'N/A')),
            'secondary_tags': str(row.get('secondary_tags', 'N/A'))
        }
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)

    print(f"Loaded {len(documents)} documents.")
    return documents

def get_pinecone_vectorstore(
    embeddings: Embeddings,
    force_reseed: bool = False
) -> Optional[PineconeVectorStore]:
    """
    Initializes Pinecone, creates the index if needed, and connects to it.
    """
    print(f"Initializing Pinecone vector store for index: '{PINECONE_INDEX_NAME}'")

    try:
        pc = Pinecone()
    except Exception as e:
        print(f"Error initializing Pinecone client: {e}")
        return None

    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Index not found. Creating new serverless index '{PINECONE_INDEX_NAME}'...")
        try:
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print("Waiting for index to be initialized...")
            while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
                time.sleep(5)
            print("Index created successfully.")
            print("Waiting 10s for new index data plane to warm up...")
            time.sleep(10)
        except Exception as e:
            print(f"Error creating Pinecone index: {e}")
            return None
    else:
        print("Found existing index.")

    index = pc.Index(PINECONE_INDEX_NAME)

    try:
        stats = index.describe_index_stats()
        vector_count = stats.get('total_vector_count', 0)

        if force_reseed and vector_count > 0:
            print(f"Forcing re-seed: Deleting all {vector_count} existing vectors...")
            index.delete(delete_all=True)
            vector_count = 0
        elif force_reseed:
            print("Forcing re-seed: Index is already empty.")

        if vector_count == 0:
            print("❌ Pinecone index is empty. Please run seed_from_jsonl.py manually before launching chat_cli.py.")
            return None
        else:
            print(f"Index already contains {vector_count} vectors. Connecting.")
            vectorstore = PineconeVectorStore.from_existing_index(
                PINECONE_INDEX_NAME,
                embeddings
            )
            return vectorstore

    except Exception as e:
        print(f"Error checking/seeding Pinecone index: {e}")
        return None



# --- Section 3: RAG Agent Orchestration (LCEL) ---
# ==================================================

def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List[BaseMessage]:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer

def _format_docs_with_citations(docs: List[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        citation_info = (
            f"Source: {doc.metadata.get('source_dataset', 'N/A')} "
            f"(Tag: {doc.metadata.get('primary_tag', 'N/A')}, "
            f"ID: {doc.metadata.get('id', 'N/A')})"
        )
        formatted_doc = (
            f"[Retrieved Document {i+1} - {citation_info}]\n"
            f"{doc.page_content}\n"
            f"[End of Document {i+1}]"
        )
        formatted_docs.append(formatted_doc)

    if not formatted_docs:
        return "No relevant documents were found."

    return "\n\n" + "\n\n".join(formatted_docs)

def create_rag_pipeline(
    retriever: BaseRetriever,
    llm: BaseChatModel
) -> Runnable:
    # --- 1. Contextualizer Chain ---
    contextualizer_prompt = ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    contextualizer_chain = (
        contextualizer_prompt
        | llm
        | StrOutputParser()
    )

    # --- 2. Main RAG Chain ---
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "Question: {question}\n\nContext: {context}\n\nAnswer:"),
    ])

    rag_chain = (
        RunnablePassthrough.assign(
            context=(lambda x: x["context"]) | RunnableLambda(_format_docs_with_citations)
        )
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # --- 3. Full Conversational RAG Pipeline ---
    def route_to_retriever(input_dict: Dict[str, Any]) -> Runnable:
        def maybe_contextualize(x):
            question = x.get("question", "").lower()
            vague_refs = ["this", "that", "above", "earlier"]
            if any(ref in question for ref in vague_refs):
                return contextualizer_chain.invoke(x)
            return x["question"]

        return (
            RunnablePassthrough.assign(
                standalone_question=maybe_contextualize
            )
            | RunnablePassthrough.assign(
                retrieved_docs=(
                    lambda x: x["standalone_question"]
                ) | retriever
            )
        )

    full_rag_pipeline = (
        route_to_retriever
        | RunnablePassthrough.assign(
            answer=(
                lambda x: {
                    "question": x["standalone_question"],
                    "context": x["retrieved_docs"]
                }
            ) | rag_chain
        )
    )

    # --- 4. Final Output Chain ---
    output_chain = (
        full_rag_pipeline
        | RunnableLambda(
            lambda x: {
                "answer": x["answer"],
                "retrieved_docs": x["retrieved_docs"]
            }
        )
    )

    return output_chain