# PennyBot: LLM Agentic RAG

**A production-ready financial question-answering chatbot using Retrieval-Augmented Generation (RAG)**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Mathematical Foundations](#mathematical-foundations)
- [Cost Analysis](#cost-analysis)
- [Evaluation Metrics](#evaluation-metrics)
- [Docker Deployment](#docker-deployment)
- [Usage Examples](#usage-examples)
- [References](#references)

---

## Overview

PennyBot is an **LLM-powered Agentic RAG system** designed for financial question-answering. It combines:

- **Dense Vector Retrieval** using Pinecone and FAISS
- **Together AI** for cost-efficient embeddings and LLM inference
- **LangChain LCEL** for orchestration and conversational context
- **Hallucination Detection** with taxonomy logging
- **TTFT & Latency Tracking** for performance monitoring

### Key Capabilities

- Answers questions on corporate finance, accounting, quantitative finance, and portfolio theory
- Retrieves relevant context from a vector database of 10,000+ financial Q&A pairs
- Cites sources with metadata for transparency
- Maintains conversational context across multi-turn interactions
- Tracks performance metrics (EM, F1, TTFT, hallucination rate)

---

## Features

✅ **Agentic RAG Pipeline** - Contextualizes queries, retrieves relevant documents, generates grounded answers
✅ **Conversational Memory** - Maintains chat history and resolves vague references
✅ **Citation Tracking** - Returns source metadata with every response
✅ **Dockerized Deployment** - Reproducible builds with GPU support
✅ **Evaluation Harness** - Automated benchmarking with EM, F1, and hallucination detection
✅ **Cost Optimized** - ~$0.05/query using Together AI and Pinecone serverless

---

## Project Structure

```
PennyBot_LLM_Agentic_RAG/
├── src/                          # Source code
│   ├── __init__.py
│   ├── rag_agent_library.py      # Core RAG orchestration (LCEL)
│   ├── chat_cli.py               # Interactive CLI interface
│   └── utils/
│       ├── __init__.py
│       └── etl.py                # Data cleaning & preprocessing
├── scripts/                      # Utility scripts
│   ├── ingest_and_filter.py      # Load and clean CSV data
│   ├── build_index.py            # Populate Pinecone index
│   ├── generate_corpus.py        # Synthetic data generation
│   └── evaluate.py               # Evaluation harness
├── data/                         # Data directory
│   ├── raw/                      # Raw datasets
│   │   ├── all_questions_tagged.csv
│   │   └── financebench_open_source.jsonl
│   └── processed/                # Cleaned data outputs
├── config/                       # Configuration
│   └── .env.example              # Environment template
├── docs/                         # Documentation
├── tests/                        # Unit tests (future)
├── .gitignore
├── .dockerignore
├── Dockerfile
├── docker-compose.yml            # Multi-service orchestration
├── requirements.txt              # Python dependencies
├── LICENSE
└── README.md
```

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- API keys from:
  - [Together AI](https://api.together.xyz/) (for embeddings & LLM)
  - [Pinecone](https://www.pinecone.io/) (for vector database)

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/PennyBot_LLM_Agentic_RAG.git
cd PennyBot_LLM_Agentic_RAG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configuration

```bash
# Copy environment template
cp config/.env.example .env

# Edit .env with your API keys
TOGETHER_API_KEY=your_together_key_here
PINECONE_API_KEY=your_pinecone_key_here
```

### 4. Build Vector Index

```bash
# Process raw data
python scripts/ingest_and_filter.py

# Build Pinecone index (one-time setup)
python scripts/build_index.py
```

### 5. Run the Chatbot

```bash
# Launch interactive CLI
python src/chat_cli.py
```

### 6. Run Evaluation

```bash
# Evaluate on benchmark dataset
python scripts/evaluate.py --dataset data/raw/all_questions_tagged.csv --limit 100

# Full evaluation (no limit)
python scripts/evaluate.py
```

---

## Mathematical Foundations

### Document Chunking

Let D = {d₁, d₂, ..., dₙ} be a dataset of documents. Each document dᵢ is segmented into smaller chunks cᵢⱼ:

**C = {c₁₁, c₁₂, ..., cₙₘ}**

### Embedding Function

Each chunk c ∈ C is mapped to a high-dimensional vector space:

**vᴄ = f(c) ∈ ℝᵈ**

Where f is the embedding model (BAAI/bge-base-en-v1.5, d=768).

### Similarity Search

Cosine similarity between query q and chunk vᴄ:

**sim(q, vᴄ) = (q · vᴄ) / (‖q‖ · ‖vᴄ‖)**

### Retrieval

Retrieve top-k most similar chunks:

**R(q) = arg topₖ sim(q, vᴄ) for c ∈ C**

### Augmented Generation

Concatenate retrieved context with query and pass to LLM:

**Answer(q) = LLM(q ⊕ R(q))**

Where ⊕ denotes concatenation.

### Weighted Context Fusion

**P(q) = q ⊕ Σᵢ₌₁ᵏ αᵢ · cᵢ**

Where αᵢ are weights based on similarity scores.

### Time to First Token (TTFT)

**TTFT = t_first - t_request**

**Total Latency = t_last - t_request**

### Hallucination Taxonomy

```
H(x) = {
  0: grounded in retrieved context
  1: unsupported numeric claim
  2: unsupported textual claim
}
```

---

## Cost Analysis

### Token Cost Function

**Cost_tokens = λ · InputTokens + μ · OutputTokens**

Where λ and μ are provider-specific rates.

### Retrieval Cost Function

**Cost_retrieval = α · k + β · Latency**

Where k is the number of retrieved chunks.

### Total Pipeline Cost

**Cost_total = Cost_tokens + Cost_retrieval + Energy_CUDA**

### Cost Estimates (Approximate)

| Component | Cost per 1K Queries |
|-----------|---------------------|
| Together AI Embeddings | ~$0.02 |
| Together AI LLM (Llama-3-70B) | ~$0.30 |
| Pinecone Storage | ~$0.25/GB/month |
| **End-to-End Pipeline** | **~$0.05/query** |

*Target: 84.5% accuracy, 100% coverage*

---

## Evaluation Metrics

The evaluation harness (`scripts/evaluate.py`) computes:

1. **Exact Match (EM)** - Binary check if normalized prediction equals ground truth
2. **Token F1** - Harmonic mean of precision/recall over token overlap
3. **TTFT** - Time to first token (ms)
4. **Total Latency** - End-to-end response time (ms)
5. **Hallucination Rate** - % grounded vs. unsupported claims
6. **Token Usage** - Input/output token counts

Results are saved to `results_tagged.csv` with per-question details.

### Sample Output

```
============================================================
EVALUATION SUMMARY
============================================================
Exact Match:            72.45%
Token F1:               84.50%
Avg Total Latency:      1,234.5 ms
Avg TTFT:               123.4 ms
Grounded Responses:     87.3%
Unsupported Numeric:    8.2%
Unsupported Claims:     4.5%
Avg Docs Retrieved:     3.0
============================================================
```

---

## Docker Deployment

### Build Docker Image

```bash
docker build -t pennybot .
```

### Run with Docker

```bash
# Basic run (evaluation mode)
docker run -it --env-file .env pennybot

# Interactive chat mode
docker run -it --env-file .env pennybot python src/chat_cli.py

# Mount volumes for data persistence
docker run -it -v $(pwd)/data:/app/data --env-file .env pennybot

# GPU acceleration (requires nvidia-docker2)
docker run --gpus all -it --env-file .env pennybot
```

### Docker Compose (Multi-Service)

```bash
# Coming soon: Redis caching + Prometheus monitoring
docker-compose up
```

---

## Usage Examples

### Interactive Chat

```
You: What is Return on Equity (ROE)?

AI: Thinking...

AI: Return on Equity (ROE) is calculated as:

ROE = Net Income ÷ Shareholder's Equity

It measures a company's profitability relative to equity invested.
Higher ROE indicates more efficient use of shareholder capital.

--- Citations ---
Source: synthetic_finance_council, ID: synthetic_00001
-----------------

You: How is it different from ROA?

AI: Thinking...

AI: ROE (Return on Equity) measures profitability relative to shareholder
equity, while ROA (Return on Assets) measures profitability relative to
total assets. Key differences:

• ROE = Net Income / Equity
• ROA = Net Income / Total Assets
• ROE reflects leverage; ROA does not

A company with high debt will have higher ROE than ROA.

--- Citations ---
Source: synthetic_finance_council, ID: synthetic_00023
-----------------
```

### Programmatic Usage

```python
from src.rag_agent_library import (
    get_pinecone_vectorstore,
    create_rag_pipeline
)
from langchain_together import ChatTogether, TogetherEmbeddings

# Initialize
llm = ChatTogether(model="meta-llama/Llama-3-70b-chat-hf", temperature=0.1)
embeddings = TogetherEmbeddings(model="BAAI/bge-base-en-v1.5")

# Get vector store
vectorstore = get_pinecone_vectorstore(embeddings)
retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

# Create RAG agent
rag_agent = create_rag_pipeline(retriever, llm)

# Query
response = rag_agent.invoke({
    "question": "How is WACC calculated?",
    "chat_history": []
})

print(response['answer'])
print(f"Retrieved {len(response['retrieved_docs'])} documents")
```

---

## References

### Core Frameworks

- **VeritasFi (2025)** - Hybrid retrieval + reranking for financial QA
- **Multi-HyDE (2025)** - Hypothetical document embeddings for multi-hop reasoning
- **FinSage (2025)** - Multi-modal retrieval with hallucination reduction
- **Financial Report Chunking (2024)** - Element-based chunking for financial docs
- **FinQANet (2022)** - Program-of-thought reasoning for financial questions

### Baselines

- **LightRAG** - Dense retrieval baseline
- **GraphRAG** - Graph-structured retrieval
- **BM25** - Sparse keyword retrieval
- **FAISS** - Facebook AI Similarity Search
- **Hybrid (BM25 + FAISS)** - Combined sparse-dense retrieval

### Statistical Methods

- **Efron & Tibshirani (1993)** - Bootstrap confidence intervals
- **Wilcoxon (1945)** - Signed-rank test for paired comparisons

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black src/ scripts/

# Lint
flake8 src/ scripts/
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

Built with:
- [LangChain](https://python.langchain.com/) - RAG orchestration
- [Pinecone](https://www.pinecone.io/) - Vector database
- [Together AI](https://www.together.ai/) - LLM inference & embeddings
- [FAISS](https://github.com/facebookresearch/faiss) - Similarity search

---

## Contact

For questions or feedback, please open an issue on GitHub.

**PennyBot** - Making financial knowledge accessible through AI 🤖📊
