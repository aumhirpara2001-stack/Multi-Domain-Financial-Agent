
---


PennyBot_LLM_Agentic_RAG

**PennyBot reborn as an LLM‑Agentic RAG Chatbot**  
Dockerized, CUDA‑accelerated, TTFT tracked, hallucination taxonomy logged, and orchestrated end‑to‑end with sustainable low‑token, low‑energy retrieval.

---

📘 Part I. Mathematical Foundations (Textbook Mode)


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
---


6. Evaluation Metrics
- **Exact Match (EM)**: binary check if normalized prediction = gold.  
- **Token F1**: harmonic mean of precision/recall over token overlap.  
- **TTFT**: time to first token.  
- **Total Latency**: end‑to‑end wall‑clock time.  
- **Hallucination Taxonomy**: {grounded, unsupported_numeric, unsupported_claim}.  


📐 Prompt Engineering Math


### Weighted Context Fusion


\[
P(q) = q \oplus \sum_{i=1}^k \alpha_i \cdot c_i
\]


- \(q\) = query  
- \(c_i\) = retrieved chunk  
- \(\alpha_i\) = weight coefficient (similarity, token budget, energy cost)

### Token + Energy Cost Function


\[
\text{Cost}(R) = \lambda \cdot \text{Tokens}(R) + \mu \cdot \text{Energy}(R)
\]



### TTFT Metric


\[
\text{TTFT} = t_{\text{first}} - t_{\text{request}}
\]




\[
\text{Latency} = t_{\text{last}} - t_{\text{request}}
\]



### Hallucination Taxonomy


\[
H(x) =
\begin{cases}
0 & \text{grounded in retrieved context} \\
1 & \text{unsupported numeric claim} \\
2 & \text{unsupported textual claim}
\end{cases}
\]



### Constraint‑Driven Prompt


\[
\text{Prompt}(q) = \text{LLM}(q \oplus R(q) \mid \text{Constraints})
\]


---

## 🔑 API Keys

To run PennyBot_LLM_Agentic_RAG you’ll need free API keys:

- [Together AI](https://api.together.xyz/) → for cost‑efficient embeddings and hosted inference
- [Pinecone](https://www.pinecone.io/) → for scalable vector database
- (Optional) Hugging Face Hub → for dataset pulls and model hosting

Add them to your `.env` file:

TOGETHER_API_KEY=your_together_key  
PINECONE_API_KEY=your_pinecone_key

---

Cost-Benefit Analysis

Yes — if this README is going to be a **saga**, it needs both the *practical links* (where to grab free API keys) and the *numerical testimony* (your end‑to‑end cost slicing). Right now it reads like a textbook, but you want it to feel like a fellowship epic: math, code, lore, and economics all braided together.


---


## 💸 End‑to‑End Cost Optimization



### 1. Token Cost Function


\[
\text{Cost}_{\text{tokens}} = \lambda \cdot \text{InputTokens} + \mu \cdot \text{OutputTokens}
\]



### 2. Retrieval Cost Function


\[
\text{Cost}_{\text{retrieval}} = \alpha \cdot k + \beta \cdot \text{Latency}
\]



### 3. Total Pipeline Cost


\[
\text{Cost}_{\text{total}} = \text{Cost}_{\text{tokens}} + \text{Cost}_{\text{retrieval}} + \text{Energy}_{\text{CUDA}}
\]


---

## 📊 Approximations - Subject to Change

- **OpenAI embeddings**: ~$0.10 per 1K queries (high‑fidelity, but pricier).  
- **Together embeddings**: ~$0.02 per 1K queries (optimized, fellowship‑grade).  
- **Pinecone storage**: ~$0.25 per GB/month (scales with corpus size).  
- **CUDA acceleration**: negligible marginal cost once GPU is provisioned.  
- **End‑to‑end pipeline**: you benchmarked ~84.5% accuracy with **100% coverage** at **< $0.05/query**.

---


## 📑 Part II. Codebook Translation (Developer Manual)

### Environment Setup
```bash
pip install langchain==0.3.7 langchain-community==0.3.7 \
            langchain-openai==0.3.7 langchain-together==0.3.7 \
            faiss-cpu python-dotenv pandas datasets scikit-learn tqdm PyYAML
````

### .env File

```
.env

TOGETHER_API_KEY=your_together_key
EMBEDDING_PROVIDER=openai
```

### Retrieval + Generation

```
python
retriever = get_retriever(index_path)
docs = retriever.retrieve(query, top_k=5)
chunks = [d.page_content for d in docs]
gen_resp = call_rag_generator(query, chunks)
```

### Evaluation Harness
- Logs EM, F1, hallucination type, complexity flag.  
- Tracks TTFT, total latency, input/output tokens.  
- Appends results to `results_tagged.csv`.  
- Prints summary averages for fellowship‑grade reproducibility.

---

## ✅ Summary
PennyBot’s resurrection is not just a chatbot. It is:
- A **CUDA‑powered, Docker‑hardened RAG pipeline**.  
- A **mathematical textbook** (Part I) and **developer codebook** (Part II).  
- A **fellowship artifact**: every eval request stamped with time, tokens, hallucination taxonomy, and reproducibility.



---

# 📚 References

## Core Frameworks
- **VeritasFi (2025)** — Hybrid retrieval + reranking for financial QA.  
  *Informed PennyBot’s hybrid retriever design (CAKC, reranker practices).*

- **Multi‑HyDE (2025)** — Hypothetical document embeddings.  
  *Inspired multi‑hop reasoning, query diversification, and recall curve tracking.*

- **FinSage (2025)** — Multi‑modal retrieval, hallucination reduction.  
  *Guided hallucination taxonomy, DPO reranker, and compliance‑critical retrieval.*

- **Financial Report Chunking for Effective RAG (2024)** — Element‑based chunking.  
  *Anchored PennyBot’s element‑aware chunking and metadata logging.*

- **FinQANet (2022)** — Program‑of‑thought reasoning.  
  *Influenced step‑by‑step reasoning and interpretable outputs.*

---

## Baselines
- **LightRAG (2022)** — Dense retrieval baseline.  
- **GraphRAG (2022)** — Graph‑structured retrieval baseline.  
- **BM25 (2009)** — Sparse retrieval baseline.  
- **FAISS (2017)** — Dense retrieval baseline.  
- **BM25 + FAISS (2019)** — Hybrid sparse‑dense baseline.  

*These baselines contextualize PennyBot’s resurrection: moving beyond dense/sparse hybrids into agentic orchestration.*

---

## Statistical Methods
- **Efron & Tibshirani (1993)** — Bootstrap confidence intervals.  
  *Used for reproducible paired comparisons.*  

- **Wilcoxon (1945)** — Signed‑rank test.  
  *Applied for nonparametric paired EM/F1 comparisons.*
  
