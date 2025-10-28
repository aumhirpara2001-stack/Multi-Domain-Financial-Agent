# PennyBot
PennyBot is a CLI-native finance assistant built around Hugging Face’s FinanceBench and FinDER. She combines benchmark analysis with conversational discovery, enabling users to explore datasets, decode financial signals, and stress-test pipelines with audit-grade precision.

---

# 📘 RAG Agent: Mathematical Foundations & Codebook

A dual‑track guide to the Retrieval‑Augmented Generation (RAG) pipeline —  
**Part I** explains the math like a textbook, **Part II** shows the code like a developer’s manual.

---

## 📑 Table of Contents
1. [Part I. Mathematical Foundations](#part-i-mathematical-foundations-textbook-style)  
   - [1. Document Representation](#1-document-representation)  
   - [2. Embedding Function](#2-embedding-function)  
   - [3. Vector Store Construction](#3-vector-store-construction)  
   - [4. Retrieval](#4-retrieval)  
   - [5. Augmented Generation](#5-augmented-generation)  
2. [Part II. Codebook Translation](#part-ii-codebook-translation-developer-manual)  
   - [1. Environment Setup](#1-environment-setup)  
   - [2. Environment Variables](#2-env-file)  
   - [3. Embedder](#3-embedderpy)  
   - [4. Application](#4-apppy)  
3. [✅ Summary](#-summary)

---

Got it—here’s the full section, all five parts, with centered “true” formulas styled for GitHub README:

---

## 📐 Mathematical Foundations

### 1. Document Chunking

Let \( D = \{d₁, d₂, \dots, dₙ\} \) be a dataset of documents. Each document \( dᵢ \) is segmented into smaller textual chunks \( cᵢⱼ \), forming a new collection:

<p align="center"><strong>C = {c₁₁, c₁₂, ..., cₙₘ}</strong></p>

---

### 2. Embedding Function

Each chunk \( c \in C \) is mapped into a high-dimensional vector space via an embedding function \( f \):

<p align="center"><strong>v<sub>c</sub> = f(c) ∈ ℝᵈ</strong></p>

---

### 3. Vector Store Construction

All chunk embeddings are stored in a FAISS index:

<p align="center"><strong>V = {v<sub>c₁</sub>, v<sub>c₂</sub>, ..., v<sub>cₖ</sub>}</strong></p>

Similarity between a query vector \( q \) and a chunk vector \( v_c \) is computed using cosine similarity:

<p align="center"><strong>sim(q, v<sub>c</sub>) = (q · v<sub>c</sub>) / (‖q‖ · ‖v<sub>c</sub>‖)</strong></p>

---

### 4. Retrieval

Given a user query \( q \), we first embed it:

<p align="center"><strong>q = f(q)</strong></p>

We then retrieve the top‑k most similar chunks:

<p align="center"><strong>R(q) = arg<sub>top‑k</sub><sub>c ∈ C</sub> sim(q, v<sub>c</sub>)</strong></p>

---

### 5. Augmented Generation

The retrieved chunks \( R(q) \) are concatenated with the query and passed to the language model:

<p align="center"><strong>Answer(q) = LLM(q ⊕ R(q))</strong></p>

Here, ⊕ denotes the concatenation of the query and its retrieved context.

---

## Part II. Codebook Translation (Developer Manual)

### 1. Environment Setup
```bash
pip install langchain==0.3.7 langchain-community==0.3.7 \
            langchain-openai==0.3.7 langchain-together==0.3.7 \
            faiss-cpu python-dotenv pandas datasets scikit-learn tqdm PyYAML streamlit
```

---

### 2. `.env` File
```dotenv
OPENAI_API_KEY=your_openai_key
TOGETHER_API_KEY=your_together_key
EMBEDDING_PROVIDER=openai
```

---

### 3. `embedder.py`
```python
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_together import TogetherEmbeddings
from langchain_community.vectorstores import FAISS

def build_vector_store(df, chunk_size=500, chunk_overlap=50):
    provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
    openai_key = os.getenv("OPENAI_API_KEY")
    together_key = os.getenv("TOGETHER_API_KEY")

    if provider == "openai":
        embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    elif provider == "together":
        embeddings = TogetherEmbeddings(
            model_name="togethercomputer/m2-bert-80M-32k-retrieval",
            together_api_key=together_key
        )
    else:
        raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {provider}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    valid_rows = df[df["context"].notnull()]

    docs = [
        Document(page_content=row["context"], metadata={"question": row["question"], "answer": row["answer"]})
        for _, row in valid_rows.iterrows()
    ]

    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore
```

---

### 4. `app.py`
```python
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from embedder import build_vector_store

load_dotenv()

# Example dataset
df = pd.DataFrame([
    {"question": "What is LangChain?", "answer": "A framework for building LLM apps.", "context": "LangChain is a framework..."},
    {"question": "What is FAISS?", "answer": "A vector database for similarity search.", "context": "FAISS is a library..."}
])

# Build vector store
vectorstore = build_vector_store(df)
retriever = vectorstore.as_retriever()

# LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Retrieval-Augmented QA
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Query loop
while True:
    q = input("Ask a question: ")
    if q.lower() in ["exit", "quit"]:
        break
    print(qa.run(q))
```

---

 Developed by Garrick Pinon as part of Algoverse AI Researcher Group and also available at: https://github.com/GarrickPinon/PennyBot


---

Also available at: https://github.com/GarrickPinon/PennyBot


