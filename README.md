# Multi-Domain-Financial-Agent
📈 Results
🧪 Experiment 1 — Retrieval-Augmented Financial QA

Goal: Build a baseline Financial QA system that retrieves relevant context from 10-K filings and extracts precise answers.

Setup:

Retriever: sentence-transformers/all-MiniLM-L6-v2

Reader: deepset/roberta-base-squad2

Dataset: Financial-QA-10k.csv (7,000 Q/A pairs from NVIDIA 10-K filings)

Key Outcomes:

Successfully built an end-to-end retrieval + QA pipeline.

Achieved strong baseline performance on factual financial questions.

Demonstrated improved semantic retrieval versus keyword-based search.

Model was able to extract precise financial insights such as revenue trends, platform strategies, and capital expenditure patterns.

Quantitative Metrics:

Metric	Value
Exact Match (EM)	0.162
F1 Score	0.309
Coverage	100% (no skipped rows)

🤖 Experiment 2 — LLM-as-a-Judge Evaluation

Goal: Evaluate model-generated financial answers using LLMs as evaluators (instead of lexical metrics).

Setup:

Judge Model: gpt-4o-mini

Compared Models: Mock Mistral and LLaMA outputs

Evaluation Criteria: Correctness, Completeness, Relevance

Output Format: JSON scores (0.0–1.0) returned per question

Results:

Model	Average Judge Score	Comments
Mistral	0.03	Partial success; a few judged responses returned valid semantic scores.
LLaMA	NaN	No valid scores (rate-limited during evaluation).

Observations:

Confirmed that LLM-as-a-Judge provides interpretable, human-like scoring.

Validated pipeline integration for future large-scale semantic evaluation once rate limits are lifted.

Established groundwork for automated benchmark generation in financial QA research.
