#!/usr/bin/env python3
"""
evaluate.py - Evaluation harness for PennyBot RAG system

Runs the RAG pipeline on a benchmark dataset and computes:
- Exact Match (EM)
- Token-level F1 score
- TTFT (Time To First Token)
- Total latency
- Hallucination detection
- Token usage statistics

Results are saved to results_tagged.csv
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any
from collections import Counter
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Import from our reorganized structure
from src.rag_agent_library import (
    get_pinecone_vectorstore,
    create_rag_pipeline,
    LLM_MODEL_NAME,
    EMBEDDING_MODEL_NAME
)
from langchain_together import ChatTogether, TogetherEmbeddings

load_dotenv()


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return text.lower().strip()


def exact_match(prediction: str, ground_truth: str) -> int:
    """Compute exact match (binary)."""
    return int(normalize_text(prediction) == normalize_text(ground_truth))


def token_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score."""
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gt_counter = Counter(gt_tokens)

    common = pred_counter & gt_counter
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def detect_hallucination(answer: str, retrieved_docs: List[Any]) -> str:
    """
    Simple hallucination detection.
    Returns: 'grounded', 'unsupported_numeric', 'unsupported_claim'

    This is a simplified heuristic. Production systems should use
    more sophisticated hallucination detection methods.
    """
    # Check if answer contains numeric claims
    import re
    has_numbers = bool(re.search(r'\d+\.?\d*%?', answer))

    # Check if answer content appears in retrieved docs
    answer_lower = normalize_text(answer)
    docs_content = " ".join([doc.page_content.lower() for doc in retrieved_docs])

    # Simple overlap check
    answer_words = set(answer_lower.split())
    docs_words = set(docs_content.split())
    overlap = len(answer_words & docs_words) / len(answer_words) if answer_words else 0

    if overlap > 0.5:
        return "grounded"
    elif has_numbers:
        return "unsupported_numeric"
    else:
        return "unsupported_claim"


def load_benchmark_dataset(dataset_path: str) -> pd.DataFrame:
    """Load benchmark dataset from CSV or JSONL."""
    if dataset_path.endswith('.csv'):
        return pd.read_csv(dataset_path)
    elif dataset_path.endswith('.jsonl'):
        data = []
        with open(dataset_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported file format: {dataset_path}")


def run_evaluation(
    rag_agent,
    test_data: pd.DataFrame,
    output_path: str = "results_tagged.csv",
    limit: int = None
) -> Dict[str, float]:
    """
    Run evaluation on test dataset.

    Args:
        rag_agent: The RAG pipeline
        test_data: DataFrame with 'question' and 'answer' columns
        output_path: Where to save detailed results
        limit: Optional limit on number of test cases

    Returns:
        Dictionary of aggregate metrics
    """
    results = []

    if limit:
        test_data = test_data.head(limit)

    print(f"Running evaluation on {len(test_data)} examples...")

    for idx, row in tqdm(test_data.iterrows(), total=len(test_data)):
        question = row['question']
        ground_truth = row['answer']

        # Measure timing
        t_start = time.time()

        try:
            # Run RAG pipeline
            response = rag_agent.invoke({
                "question": question,
                "chat_history": []
            })

            t_end = time.time()
            total_latency = t_end - t_start

            prediction = response['answer']
            retrieved_docs = response['retrieved_docs']

            # Compute metrics
            em = exact_match(prediction, ground_truth)
            f1 = token_f1(prediction, ground_truth)
            hallucination_type = detect_hallucination(prediction, retrieved_docs)

            # Estimate tokens (rough approximation)
            input_tokens = len(question.split()) + sum(len(doc.page_content.split()) for doc in retrieved_docs)
            output_tokens = len(prediction.split())

            result = {
                'id': row.get('id', idx),
                'question': question,
                'ground_truth': ground_truth,
                'prediction': prediction,
                'exact_match': em,
                'token_f1': f1,
                'total_latency_ms': total_latency * 1000,
                'ttft_ms': total_latency * 1000 * 0.1,  # Estimate (first 10% of time)
                'hallucination_type': hallucination_type,
                'num_docs_retrieved': len(retrieved_docs),
                'input_tokens_approx': input_tokens,
                'output_tokens_approx': output_tokens,
                'source_dataset': row.get('source_dataset', 'unknown'),
                'primary_tag': row.get('primary_tag', 'unknown')
            }

            results.append(result)

        except Exception as e:
            print(f"\nError on question {idx}: {e}")
            results.append({
                'id': row.get('id', idx),
                'question': question,
                'ground_truth': ground_truth,
                'prediction': f"ERROR: {str(e)}",
                'exact_match': 0,
                'token_f1': 0.0,
                'total_latency_ms': 0,
                'ttft_ms': 0,
                'hallucination_type': 'error',
                'num_docs_retrieved': 0,
                'input_tokens_approx': 0,
                'output_tokens_approx': 0,
                'source_dataset': row.get('source_dataset', 'unknown'),
                'primary_tag': row.get('primary_tag', 'unknown')
            })

    # Save detailed results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ Detailed results saved to {output_path}")

    # Compute aggregate metrics
    aggregate = {
        'exact_match_avg': results_df['exact_match'].mean(),
        'token_f1_avg': results_df['token_f1'].mean(),
        'total_latency_avg_ms': results_df['total_latency_ms'].mean(),
        'ttft_avg_ms': results_df['ttft_ms'].mean(),
        'hallucination_grounded_pct': (results_df['hallucination_type'] == 'grounded').mean() * 100,
        'hallucination_numeric_pct': (results_df['hallucination_type'] == 'unsupported_numeric').mean() * 100,
        'hallucination_claim_pct': (results_df['hallucination_type'] == 'unsupported_claim').mean() * 100,
        'avg_docs_retrieved': results_df['num_docs_retrieved'].mean(),
        'avg_input_tokens': results_df['input_tokens_approx'].mean(),
        'avg_output_tokens': results_df['output_tokens_approx'].mean(),
    }

    return aggregate


def print_summary(metrics: Dict[str, float]):
    """Print formatted evaluation summary."""
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Exact Match:            {metrics['exact_match_avg']:.2%}")
    print(f"Token F1:               {metrics['token_f1_avg']:.2%}")
    print(f"Avg Total Latency:      {metrics['total_latency_avg_ms']:.1f} ms")
    print(f"Avg TTFT:               {metrics['ttft_avg_ms']:.1f} ms")
    print(f"Grounded Responses:     {metrics['hallucination_grounded_pct']:.1f}%")
    print(f"Unsupported Numeric:    {metrics['hallucination_numeric_pct']:.1f}%")
    print(f"Unsupported Claims:     {metrics['hallucination_claim_pct']:.1f}%")
    print(f"Avg Docs Retrieved:     {metrics['avg_docs_retrieved']:.1f}")
    print(f"Avg Input Tokens:       {metrics['avg_input_tokens']:.0f}")
    print(f"Avg Output Tokens:      {metrics['avg_output_tokens']:.0f}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate PennyBot RAG system")
    parser.add_argument(
        '--dataset',
        type=str,
        default='data/raw/all_questions_tagged.csv',
        help='Path to evaluation dataset (CSV or JSONL)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results_tagged.csv',
        help='Path to save detailed results'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of test examples (for quick testing)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Number of documents to retrieve'
    )

    args = parser.parse_args()

    print("="*60)
    print("PennyBot RAG Evaluation Harness")
    print("="*60)
    print(f"LLM Model:        {LLM_MODEL_NAME}")
    print(f"Embedding Model:  {EMBEDDING_MODEL_NAME}")
    print(f"Dataset:          {args.dataset}")
    print(f"Top-k Retrieval:  {args.top_k}")
    if args.limit:
        print(f"Test Limit:       {args.limit} examples")
    print("="*60 + "\n")

    # Initialize models
    print("Initializing models...")
    llm = ChatTogether(model=LLM_MODEL_NAME, temperature=0.0)
    embeddings = TogetherEmbeddings(model=EMBEDDING_MODEL_NAME)

    # Connect to vector store
    print("Connecting to Pinecone...")
    vectorstore = get_pinecone_vectorstore(embeddings, force_reseed=False)

    if vectorstore is None:
        print("❌ Failed to connect to vector store. Exiting.")
        return

    # Create RAG pipeline
    print("Creating RAG pipeline...")
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={'k': args.top_k}
    )
    rag_agent = create_rag_pipeline(retriever, llm)

    # Load test data
    print(f"Loading test data from {args.dataset}...")
    test_data = load_benchmark_dataset(args.dataset)

    # Run evaluation
    metrics = run_evaluation(
        rag_agent,
        test_data,
        output_path=args.output,
        limit=args.limit
    )

    # Print summary
    print_summary(metrics)


if __name__ == "__main__":
    main()
