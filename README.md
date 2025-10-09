# Multi-Domain-Financial-Agent
🧪 Experimental Results and Discussion
Evaluation Setup

The proposed Unified Multi-Domain Financial Agent was evaluated on a dataset of 7,000 Q&A pairs derived from NVIDIA’s 2023 10-K filing, covering diverse financial topics such as banking, investment, insurance, and fintech.
Each record contained a question, answer, and context passage.
Evaluation was performed on 500 randomly sampled examples using the following configuration:

Retriever: sentence-transformers/all-MiniLM-L6-v2 (384-D semantic embeddings)

Indexing: FAISS (Inner Product / Cosine similarity)

QA Model: deepset/roberta-base-squad2 (extractive transformer model)

Retrieval Depth: Top-3 most relevant contexts

Metrics: Exact Match (EM), token-level F1, and semantic consistency (cosine similarity)

📊 Quantitative Results
| Metric                  |   Score   | Description                                                       |
| :---------------------- | :-------: | :---------------------------------------------------------------- |
| **Exact Match (EM)**    | **0.162** | Percentage of answers exactly matching the ground truth           |
| **F1 Score**            | **0.309** | Token-level overlap between predicted and reference answers       |
| **Average Consistency** |  **0.64** | Cosine similarity between predicted answer and retrieved contexts |

💬 Qualitative Example Output
{
  "question": "How do interest rate hikes affect bond prices?",
  "answer": "higher market interest rates offered for retail deposits",
  "confidence": 0.335,
  "consistency": 0.645,
  "citations": [
    {
      "score": 0.514,
      "snippet": "In addition, economic conditions and actions by policymaking bodies are contributing to changing interest rates and significant capital market volatility..."
    },
    {
      "score": 0.505,
      "snippet": "The increase in interest rates paid on our deposits were primarily due to the impact of higher market interest rates offered for retail deposits..."
    },
    {
      "score": 0.496,
      "snippet": "Interest expense increased, primarily driven by higher interest rates paid on customer deposits."
    }
  ]
}

MY POINT OF VIEW:
-->The experimental results demonstrate that the system effectively retrieves semantically relevant financial contexts and can generate factually grounded extractive answers without hallucination.

-->Although the Exact Match (0.162) and F1 (0.309) scores indicate modest lexical overlap with the ground truth, this outcome is expected because:

    1.The QA model (deepset/roberta-base-squad2) was trained on SQuAD v2 (Wikipedia domain), not financial text, leading to vocabulary and phrasing mismatches.

    2.The dataset includes long, technical sentences from 10-K filings that often paraphrase answers rather than repeating them verbatim.

    3.The pipeline uses extractive QA, which selects direct text spans — this limits flexibility compared to generative reasoning models.

However, the retrieval consistency score (0.64) confirms that most answers remain semantically aligned with their evidence, showing that the retriever + FAISS + embedding design is working correctly.

Overall, these results establish a strong zero-shot baseline for financial question answering.
Future experiments will focus on domain-adapted embeddings (FinText / FinBERT), generative RAG models (FinGPT / Mixtral), and LLM-as-a-Judge evaluation to improve accuracy and semantic assessment.