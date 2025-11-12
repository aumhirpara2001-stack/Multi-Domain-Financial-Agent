"""Evaluation metrics for comparing predicted and gold answers."""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any


def normalize_text(text: str) -> str:
    """Normalize text for comparison.

    Args:
        text: Input text

    Returns:
        str: Normalized text (lowercase, stripped, single spaces)
    """
    if not text or pd.isna(text):
        return ""

    # Convert to string and lowercase
    text = str(text).lower().strip()

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove punctuation at start/end
    text = text.strip('.,!?;:')

    return text


# def exact_match(predicted: str, gold: str) -> int:
#     """Calculate exact match score (1 or 0).

#     DEPRECATED: Exact match is too strict for most QA tasks.
#     Use semantic_similarity instead for more meaningful evaluation.

#     Args:
#         predicted: Predicted answer
#         gold: Gold standard answer

#     Returns:
#         int: 1 if exact match after normalization, 0 otherwise
#     """
#     pred_norm = normalize_text(predicted)
#     gold_norm = normalize_text(gold)

#     return 1 if pred_norm == gold_norm else 0


def embedding_similarity(predicted: str, gold: str, embeddings) -> float:
    """Calculate semantic similarity using embeddings.

    Args:
        predicted: Predicted answer
        gold: Gold standard answer
        embeddings: Embeddings model from LangChain (TogetherEmbeddings)

    Returns:
        float: Cosine similarity score [0, 1]
    """
    if not predicted or not gold or pd.isna(predicted) or pd.isna(gold):
        return 0.0

    try:
        # Embed both texts
        pred_embedding = embeddings.embed_query(str(predicted))
        gold_embedding = embeddings.embed_query(str(gold))

        # Calculate cosine similarity
        pred_vec = np.array(pred_embedding)
        gold_vec = np.array(gold_embedding)

        # Cosine similarity: dot product / (norm1 * norm2)
        similarity = np.dot(pred_vec, gold_vec) / (
            np.linalg.norm(pred_vec) * np.linalg.norm(gold_vec)
        )

        # Clip to [0, 1] range (in case of numerical issues)
        similarity = np.clip(similarity, 0.0, 1.0)

        return float(similarity)
        
    except Exception as e:
        print(f"Warning: Failed to calculate embedding similarity: {str(e)}")
        return 0.0


def calculate_aggregate_metrics(results_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate aggregate metrics across all results.

    Args:
        results_df: DataFrame with columns: semantic_similarity,
                   question_type, retrieval_time_ms, generation_time_ms, error

    Returns:
        Dict containing:
            - avg_semantic_similarity: Average semantic similarity
            - min_semantic_similarity: Minimum semantic similarity
            - max_semantic_similarity: Maximum semantic similarity
            - similarity_by_type: Dict of similarity per question type
            - avg_retrieval_time_ms: Average retrieval time
            - avg_generation_time_ms: Average generation time
            - total_questions: Total number of questions
            - successful_predictions: Number of non-null predictions
            - failed_predictions: Number of failed predictions
            - success_rate: Percentage of successful predictions
    """
    metrics = {}

    # Filter out failed predictions (those with errors)
    successful = results_df[results_df['error'].isna()]
    failed = results_df[results_df['error'].notna()]

    total = len(results_df)
    num_successful = len(successful)
    num_failed = len(failed)

    metrics['total_questions'] = total
    metrics['successful_predictions'] = num_successful
    metrics['failed_predictions'] = num_failed
    metrics['success_rate'] = (num_successful / total * 100) if total > 0 else 0.0

    # Calculate metrics only on successful predictions
    if num_successful > 0:
        # Semantic similarity statistics
        metrics['avg_semantic_similarity'] = successful['semantic_similarity'].mean()
        metrics['min_semantic_similarity'] = successful['semantic_similarity'].min()
        metrics['max_semantic_similarity'] = successful['semantic_similarity'].max()

        # Similarity by question type
        if 'question_type' in successful.columns:
            similarity_by_type = successful.groupby('question_type')['semantic_similarity'].agg(['mean', 'count', 'min', 'max'])
            metrics['similarity_by_type'] = similarity_by_type.to_dict('index')
        else:
            metrics['similarity_by_type'] = {}

        # Timing metrics
        if 'retrieval_time_ms' in successful.columns:
            metrics['avg_retrieval_time_ms'] = successful['retrieval_time_ms'].mean()

        if 'generation_time_ms' in successful.columns:
            metrics['avg_generation_time_ms'] = successful['generation_time_ms'].mean()
    else:
        # No successful predictions
        metrics['avg_semantic_similarity'] = 0.0
        metrics['min_semantic_similarity'] = 0.0
        metrics['max_semantic_similarity'] = 0.0
        metrics['similarity_by_type'] = {}
        metrics['avg_retrieval_time_ms'] = 0.0
        metrics['avg_generation_time_ms'] = 0.0

    return metrics


def format_metrics_summary(metrics: Dict[str, Any]) -> str:
    """Format metrics dictionary as a readable string.

    Args:
        metrics: Metrics dictionary from calculate_aggregate_metrics

    Returns:
        str: Formatted metrics summary
    """
    lines = [
        "\n" + "="*60,
        "BULK TEST RESULTS SUMMARY",
        "="*60,
        f"\nTotal Questions: {metrics['total_questions']}",
        f"Successful Predictions: {metrics['successful_predictions']}",
        f"Failed Predictions: {metrics['failed_predictions']}",
        f"Success Rate: {metrics['success_rate']:.2f}%",
        "",
        "SEMANTIC SIMILARITY METRICS:",
        f"  Average: {metrics['avg_semantic_similarity']:.4f}",
        f"  Min: {metrics['min_semantic_similarity']:.4f}",
        f"  Max: {metrics['max_semantic_similarity']:.4f}",
        "",
    ]

    # Add timing info if available
    if metrics.get('avg_retrieval_time_ms', 0) > 0:
        lines.append(f"Average Retrieval Time: {metrics['avg_retrieval_time_ms']:.1f} ms")

    if metrics.get('avg_generation_time_ms', 0) > 0:
        lines.append(f"Average Generation Time: {metrics['avg_generation_time_ms']:.1f} ms")

    # Add similarity by question type
    if metrics.get('similarity_by_type'):
        lines.append("\nSemantic Similarity by Question Type:")
        for qtype, stats in metrics['similarity_by_type'].items():
            lines.append(f"  {qtype}:")
            lines.append(f"    Mean: {stats['mean']:.4f} | Min: {stats['min']:.4f} | Max: {stats['max']:.4f} | Count: {int(stats['count'])}")

    lines.append("="*60 + "\n")

    return "\n".join(lines)
