"""Bulk testing framework for RAG system evaluation."""

import os
import sys
import time
import json
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
import openai
from dotenv import load_dotenv
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import retrieval components
from langchain_chroma import Chroma
from langchain_together import TogetherEmbeddings

# Import custom modules
from dataset_adapters import BaseDatasetAdapter, FinanceBenchAdapter
from evaluation.metrics import (
    embedding_similarity,
    calculate_aggregate_metrics,
    format_metrics_summary
)
from evaluation.llm_judge import llm_as_judge

# Load environment variables
load_dotenv()


@dataclass
class BulkTestConfig:
    """Configuration for bulk testing runs."""

    # Dataset settings
    dataset_name: str

    # Model settings
    model_name: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    embedding_model: str = "text-embedding-3-large"

    # Retrieval settings
    top_k_retrieval: int = 5
    use_hybrid_search: bool = True  # BM25 + Semantic
    use_metadata_filtering: bool = True  # Filter by company/year
    use_reranking: bool = True  # Cross-encoder reranking

    # Generation settings
    temperature: float = 0.0
    max_tokens: int = 512

    # Evaluation settings
    use_llm_judge: bool = False
    judge_model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"

    # Paths
    chroma_path: str = "chroma"
    output_dir: str = "bulk_runs"

    # Runtime metadata
    timestamp: str = None

    def __post_init__(self):
        """Generate timestamp if not provided and resolve paths relative to project root."""
        if self.timestamp is None:
            self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Resolve paths relative to project root
        base_dir = Path(__file__).parent.parent
        self.chroma_path = str(base_dir / self.chroma_path)
        self.output_dir = str(base_dir / self.output_dir)

    def get_model_abbrev(self) -> str:
        """Get abbreviated model name for filename."""
        if "llama-3.1-70b" in self.model_name.lower():
            return "llama31-70b"
        elif "llama-3.1-8b" in self.model_name.lower():
            return "llama31-8b"
        else:
            # Generic abbreviation
            return self.model_name.split("/")[-1][:20]

    def generate_filename(self, dataset_abbrev: str) -> str:
        """Generate output filename from configuration.

        Format: {timestamp}_{dataset}_{model}_k{top_k}_t{temp}.csv

        Args:
            dataset_abbrev: Short name for dataset (e.g., 'fb')

        Returns:
            str: Generated filename
        """
        model_abbrev = self.get_model_abbrev()
        temp_str = f"t{self.temperature}".replace(".", "")

        filename = (
            f"{self.timestamp}_{dataset_abbrev}_{model_abbrev}_"
            f"k{self.top_k_retrieval}_{temp_str}.csv"
        )

        return filename


class BulkTestRunner:
    """Main bulk testing runner."""

    def __init__(self, config: BulkTestConfig):
        """Initialize bulk test runner.

        Args:
            config: Test configuration
        """
        self.config = config
        self.retriever = None
        self.embeddings = None
        self.llm_client = None

    def initialize_framework(self):
        """Initialize RAG framework components.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            print("\nInitializing RAG framework...")

            # Initialize embeddings
            print(f"  Loading embeddings model: {self.config.embedding_model}")
            from langchain_openai import OpenAIEmbeddings
            self.embeddings = OpenAIEmbeddings(model=self.config.embedding_model)

            # Load ChromaDB
            print(f"  Loading ChromaDB from: {self.config.chroma_path}")
            db = Chroma(
                persist_directory=self.config.chroma_path,
                embedding_function=self.embeddings
            )

            # Create retriever based on config
            if self.config.use_hybrid_search:
                print(f"  Creating HYBRID retriever (BM25 + Semantic, top_k={self.config.top_k_retrieval})")

                # Get all documents for BM25
                all_docs = db.get()
                from langchain_core.documents import Document
                documents = [
                    Document(page_content=text, metadata=meta)
                    for text, meta in zip(all_docs['documents'], all_docs['metadatas'])
                ]

                # Create BM25 retriever
                from langchain.retrievers import BM25Retriever
                bm25_retriever = BM25Retriever.from_documents(documents)
                bm25_retriever.k = self.config.top_k_retrieval

                # Create semantic retriever
                semantic_retriever = db.as_retriever(
                    search_kwargs={"k": self.config.top_k_retrieval}
                )

                # Combine with ensemble (50% weight each)
                from langchain.retrievers import EnsembleRetriever
                self.retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, semantic_retriever],
                    weights=[0.5, 0.5]
                )
            else:
                print(f"  Creating SEMANTIC-ONLY retriever (top_k={self.config.top_k_retrieval})")
                self.retriever = db.as_retriever(
                    search_kwargs={"k": self.config.top_k_retrieval}
                )

            # Initialize LLM client (Together API via OpenAI SDK)
            print(f"  Initializing LLM client: {self.config.model_name}")
            self.llm_client = openai.OpenAI(
                api_key=os.environ.get("TOGETHER_API_KEY"),
                base_url="https://api.together.xyz/v1",
            )

            print("Framework initialization complete!\n")
            return True

        except Exception as e:
            print(f"ERROR: Framework initialization failed: {str(e)}")
            return False

    def process_single_question(self, question: str, question_id: Any) -> Dict[str, Any]:
        """Process a single question through the RAG pipeline.

        Args:
            question: Question text
            question_id: Question identifier (for logging)

        Returns:
            Dict containing:
                - predicted_answer: Generated answer (or None if failed)
                - sources: List of source document metadata
                - retrieval_time_ms: Time for retrieval
                - generation_time_ms: Time for generation
                - error: Error message if failed, None otherwise
        """
        result = {
            'predicted_answer': None,
            'sources': None,
            'retrieval_time_ms': 0,
            'generation_time_ms': 0,
            'error': None
        }

        try:
            # Retrieval phase with optional enhancements
            retrieval_start = time.time()

            # Retrieve MORE chunks if using filtering/reranking (so we have candidates after filtering)
            retrieval_multiplier = 3 if (self.config.use_metadata_filtering or self.config.use_reranking) else 1
            initial_k = self.config.top_k_retrieval * retrieval_multiplier

            # Update retriever k values dynamically based on whether we need more chunks
            if self.config.use_hybrid_search:
                # Update both BM25 and semantic retrievers in the ensemble
                self.retriever.retrievers[0].k = initial_k  # BM25
                self.retriever.retrievers[1].search_kwargs["k"] = initial_k  # Semantic
            else:
                # Update semantic-only retriever
                self.retriever.search_kwargs["k"] = initial_k

            # Retrieve chunks
            docs = self.retriever.invoke(question)

            # Apply metadata filtering if enabled
            if self.config.use_metadata_filtering:
                from src.metadata_utils import extract_metadata_from_question, filter_chunks_by_metadata
                question_metadata = extract_metadata_from_question(question)
                filtered_docs = filter_chunks_by_metadata(docs, question_metadata)

                # If we extracted metadata (years/companies), ALWAYS use filtered results
                # Better to have few correct chunks than many contaminated ones
                if question_metadata['years'] or question_metadata['companies']:
                    if filtered_docs:
                        docs = filtered_docs  # Use filtered results, even if only 1-2 chunks
                    else:
                        # Metadata was extracted but nothing matched - this is a real "no answer" scenario
                        docs = []
                # If no metadata in question, skip filtering (e.g., "What is the revenue?" with no year/company)

            # Apply reranking if enabled and we have more than needed
            if self.config.use_reranking and len(docs) > 0:
                from sentence_transformers import CrossEncoder
                # Use a cross-encoder model for reranking
                if not hasattr(self, 'reranker'):
                    print("  Loading cross-encoder for reranking...")
                    self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

                # Score each doc against the question
                pairs = [[question, doc.page_content] for doc in docs]
                scores = self.reranker.predict(pairs)

                # Sort by score and take top k
                scored_docs = list(zip(docs, scores))
                scored_docs.sort(key=lambda x: x[1], reverse=True)
                docs = [doc for doc, score in scored_docs[:self.config.top_k_retrieval]]
            else:
                # No reranking - just take top k
                docs = docs[:self.config.top_k_retrieval]

            retrieval_time = (time.time() - retrieval_start) * 1000  # Convert to ms

            result['retrieval_time_ms'] = retrieval_time

            # Check if documents found
            if not docs:
                result['error'] = "No relevant documents found"
                return result

            # Extract context and sources
            context = "\n\n".join(d.page_content for d in docs)
            sources = [doc.metadata for doc in docs]
            result['sources'] = sources

            # Generation phase
            generation_start = time.time()

            response = self.llm_client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise financial analysis assistant. Answer questions using the information provided in the context. Be accurate with numbers, dates, and company names. ALWAYS provide your best answer based on the available context - never refuse to answer or say you cannot find the information."
                    },
                    {
                        "role": "user",
                        "content": f"""Answer the following question using the information from the provided context.

IMPORTANT INSTRUCTIONS:
- ALWAYS provide an answer - even if the context seems incomplete, give your best hypothesis based on available information
- Use precise numbers, dates, and company names from the context when available
- Do NOT use information from other companies or fiscal years unless explicitly asked
- Pay close attention to fiscal years and time periods mentioned in both the question and context
- For numerical questions requiring a specific number, percentage, or ratio as the answer:
  * Provide ONLY the numerical value with appropriate units
  * Format examples: "$1,577 million" or "65.4%" or "24.26"
  * Do NOT add explanatory sentences like "The answer is..." or "According to the context..."
- For non-numerical or explanatory questions, provide full context and reasoning
- NEVER say "The provided context does not contain sufficient information" - always attempt an answer

Context:
{context}

Question: {question}

Answer:""",
                    },
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            generation_time = (time.time() - generation_start) * 1000  # Convert to ms
            result['generation_time_ms'] = generation_time

            # Extract answer
            if response and response.choices:
                result['predicted_answer'] = response.choices[0].message.content
            else:
                result['error'] = "Empty response from LLM"

        except openai.RateLimitError as e:
            result['error'] = f"Rate limit exceeded: {str(e)}"
            # Re-raise to trigger graceful exit
            raise

        except (openai.APIConnectionError, openai.Timeout) as e:
            result['error'] = f"API connection error: {str(e)}"

        except Exception as e:
            result['error'] = f"Unexpected error: {str(e)}"

        return result

    def run_bulk_test(self, adapter: BaseDatasetAdapter) -> pd.DataFrame:
        """Run bulk test on a dataset.

        Args:
            adapter: Dataset adapter instance

        Returns:
            pd.DataFrame: Results with predictions and metrics

        Raises:
            KeyboardInterrupt: If user interrupts
            Exception: If critical error occurs
        """
        print("\n" + "="*60)
        print("STARTING BULK TEST")
        print("="*60)

        # Load dataset
        try:
            df = adapter.load_dataset()
        except Exception as e:
            print(f"ERROR: Failed to load dataset: {str(e)}")
            sys.exit(1)

        # Get column names
        question_col = adapter.get_question_column()
        answer_col = adapter.get_answer_column()
        question_type_col = adapter.get_question_type_column()
        metadata_cols = adapter.get_metadata_columns()

        print(f"Dataset loaded: {len(df)} questions")
        print(f"Question column: {question_col}")
        print(f"Answer column: {answer_col}")
        if question_type_col:
            print(f"Question type column: {question_type_col}")
        print(f"Metadata columns: {metadata_cols}")

        # Initialize framework
        if not self.initialize_framework():
            print("ERROR: Framework initialization failed. Exiting.")
            sys.exit(1)

        # Prepare results storage
        results = []

        # Process questions with progress bar
        print("\nProcessing questions...")
        start_time = time.time()

        try:
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Questions"):
                question = row[question_col]
                gold_answer = row[answer_col]

                # Process question
                result = self.process_single_question(question, idx)

                # Calculate semantic similarity if prediction succeeded
                if result['predicted_answer'] is not None:
                    sem_sim = embedding_similarity(
                        result['predicted_answer'],
                        gold_answer,
                        self.embeddings
                    )
                else:
                    sem_sim = 0.0

                # Calculate LLM judge score if enabled
                judge_score = None
                judge_justification = None
                if self.config.use_llm_judge and result['predicted_answer'] is not None:
                    judge_score, judge_justification = llm_as_judge(
                        question=question,
                        gold_answer=gold_answer,
                        predicted_answer=result['predicted_answer'],
                        judge_model=self.config.judge_model
                    )

                # Format sources for CSV
                sources_str = None
                if result['sources']:
                    source_names = [
                        s.get('source', 'unknown') for s in result['sources']
                    ]
                    sources_str = "; ".join(source_names)

                # Build result row
                result_row = {
                    'question_id': idx,
                    'question': question,
                    'gold_answer': gold_answer,
                    'predicted_answer': result['predicted_answer'],
                    'semantic_similarity': sem_sim,
                    'retrieval_time_ms': result['retrieval_time_ms'],
                    'generation_time_ms': result['generation_time_ms'],
                    'sources': sources_str,
                    'error': result['error']
                }

                # Add LLM judge scores if enabled
                if self.config.use_llm_judge:
                    result_row['judge_score'] = judge_score
                    result_row['judge_justification'] = judge_justification

                # Add question type if available
                if question_type_col and question_type_col in row:
                    result_row['question_type'] = row[question_type_col]

                # Add metadata columns
                for col in metadata_cols:
                    if col in row:
                        result_row[col] = row[col]

                results.append(result_row)

        except openai.RateLimitError as e:
            print(f"\n\nERROR: API rate limit exceeded!")
            print(f"Processed {len(results)}/{len(df)} questions before hitting limit.")
            print("Saving partial results and exiting gracefully...")

            # Save partial results
            if results:
                results_df = pd.DataFrame(results)
                self._save_results(results_df, adapter, partial=True)

            sys.exit(0)

        except KeyboardInterrupt:
            print(f"\n\nInterrupted by user!")
            print(f"Processed {len(results)}/{len(df)} questions.")
            print("Saving partial results...")

            # Save partial results
            if results:
                results_df = pd.DataFrame(results)
                self._save_results(results_df, adapter, partial=True)

            sys.exit(0)

        # Calculate total runtime
        total_time = time.time() - start_time

        print(f"\nProcessing complete!")
        print(f"Total time: {total_time:.2f} seconds")

        # Convert to DataFrame
        results_df = pd.DataFrame(results)

        return results_df

    def _save_results(self, results_df: pd.DataFrame, adapter: BaseDatasetAdapter, partial: bool = False):
        """Save results to CSV and summary to JSON.

        Args:
            results_df: Results DataFrame
            adapter: Dataset adapter (for name)
            partial: Whether this is a partial result
        """
        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)

        # Generate filename
        filename = self.config.generate_filename(adapter.get_dataset_name())

        if partial:
            filename = filename.replace('.csv', '_PARTIAL.csv')

        output_path = output_dir / filename

        # Save CSV
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

        # Calculate and save metrics
        metrics = calculate_aggregate_metrics(results_df)

        # Add config to metrics
        metrics['config'] = asdict(self.config)

        # Save summary JSON
        summary_path = output_path.with_suffix('.json')
        with open(summary_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"Summary saved to: {summary_path}")

        # Print formatted summary
        print(format_metrics_summary(metrics))

    def save_results(self, results_df: pd.DataFrame, adapter: BaseDatasetAdapter):
        """Public method to save results."""
        self._save_results(results_df, adapter, partial=False)


def main():
    """Main entry point for bulk testing."""
    parser = argparse.ArgumentParser(
        description="Run bulk testing on RAG framework"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='financebench',
        help='Dataset to test on (default: financebench)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
        help='LLM model name'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of documents to retrieve (default: 5)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Temperature for generation (default: 0.0)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=512,
        help='Max tokens for generation (default: 512)'
    )
    parser.add_argument(
        '--subset',
        type=str,
        default=None,
        help='Path to subset questions CSV (optional, filters to subset of questions)'
    )
    parser.add_argument(
        '--use-llm-judge',
        action='store_true',
        help='Enable LLM-as-a-Judge evaluation (slower but more accurate)'
    )
    parser.add_argument(
        '--judge-model',
        type=str,
        default='meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
        help='Model to use for LLM judge evaluation'
    )

    args = parser.parse_args()

    # Create configuration
    config = BulkTestConfig(
        dataset_name=args.dataset,
        model_name=args.model,
        top_k_retrieval=args.top_k,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        use_llm_judge=args.use_llm_judge,
        judge_model=args.judge_model
    )

    # Select dataset adapter
    if args.dataset.lower() == 'financebench':
        adapter = FinanceBenchAdapter(subset_csv=args.subset)
    else:
        print(f"ERROR: Unknown dataset '{args.dataset}'")
        print("Available datasets: financebench")
        sys.exit(1)

    # Create runner and execute
    runner = BulkTestRunner(config)

    try:
        results_df = runner.run_bulk_test(adapter)
        runner.save_results(results_df, adapter)

    except Exception as e:
        print(f"\nFATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
