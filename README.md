# Multi-Domain-Financial-Agent
Result:
Dataset: https://www.kaggle.com/datasets/yousefsaeedian/financial-q-and-a-10k/data
Experiment 1 — Retrieval-Augmented Financial Question Answering
—>In this experiment, I tried to developed a retrieval-augmented QA pipeline to Evaluate how well lightweight transformer models can extract factual answers from long-form financial documents (10-K filings). —>I combined two complementary models:
  * Retriever: all-MiniLM-L6-v2 (Sentence Transformer) This used to generate dense embeddings for both questions and filing paragraphs, enabling semantic similarity search via FAISS.
  * Reader: deepset/roberta-base-squad2 This used to extract the most likely answer span from the retrieved top-k paragraphs.
—>I measured standard QA metrics — Exact Match (EM) and F1 score — using the ground truth answers provided in the dataset.
Results:
  * Exact Match (EM): 0.162
  * F1 Score: 0.309
  * Observations: The model performed well on direct factual queries (“What year did NVIDIA invent the GPU?”) but struggled with multi-hop reasoning or questions requiring numeric inference. Despite its simplicity, this pipeline established a solid retrieval-based baseline with full interpretability and efficient inference.

Experiment 2 — LLM-as-a-Judge Semantic Evaluation
—>In this experiment, I performed  LLM-as-a-Judge evaluation, an emerging method where a large model is used to assess the semantic correctness of other models’ outputs.
I used GPT-4o-mini as a semantic evaluator to rate model answers on correctness, completeness, and relevance.
—>I compared two baseline generative models Mistral and LLaMA — both fine-tuned for text generation but not domain-specific to finance.
—>Each model generated answers for a subset of 50 financial questions, and GPT-4o-mini scored each output between 0.0 and 1.0 based on how closely it matched the true answer.Before 50 questions I tried with 10 questions but I got nan (incomplete) Judge Score for LLAMA due to rate limits then I tried for more questions after solved API issue but still got rate limits.
Results:
  Model	    Average                             Judge Score	Evaluation Method
  Mistral	  0.033	                              GPT-4o-mini Semantic Scoring
  LLaMA	    nan (incomplete due to rate limits)	GPT-4o-mini Semantic Scoring
—>Overall, Mistral’s average semantic score (≈0.03) indicates very low factual consistency, confirming that generic LLMs often fail on specialized financial reasoning without retrieval grounding.LLaMA’s evaluation was partially incomplete due to API rate limits, but preliminary responses showed similar trends.
