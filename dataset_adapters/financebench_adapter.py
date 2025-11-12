"""FinanceBench dataset adapter."""

from typing import Optional, List
import pandas as pd
from datasets import load_dataset
from .base_adapter import BaseDatasetAdapter


class FinanceBenchAdapter(BaseDatasetAdapter):
    """Adapter for the PatronusAI/financebench dataset from HuggingFace.

    Dataset info:
    - 150 questions about financial documents
    - Question types: information extraction, numerical reasoning, logical reasoning
    - Includes gold answers and evidence from source documents
    """

    def __init__(self, split: str = "train"):
        """Initialize FinanceBench adapter.

        Args:
            split: Dataset split to load (default: 'train')
        """
        self.split = split
        self.dataset_id = "PatronusAI/financebench"

    def load_dataset(self) -> pd.DataFrame:
        """Load FinanceBench dataset from HuggingFace.

        Returns:
            pd.DataFrame: Dataset with all columns

        Raises:
            Exception: If dataset fails to load
        """
        try:
            print(f"Loading {self.dataset_id} dataset from HuggingFace...")
            dataset = load_dataset(self.dataset_id, split=self.split)
            df = dataset.to_pandas()
            print(f"Successfully loaded {len(df)} questions from FinanceBench")
            return df
        except Exception as e:
            raise Exception(f"Failed to load FinanceBench dataset: {str(e)}")

    def get_question_column(self) -> str:
        """Return the column name for questions.

        Returns:
            str: 'question'
        """
        return "question"

    def get_answer_column(self) -> str:
        """Return the column name for gold answers.

        Returns:
            str: 'answer'
        """
        return "answer"

    def get_question_type_column(self) -> Optional[str]:
        """Return the column name for question types.

        Returns:
            str: 'question_type'
        """
        return "question_type"

    def get_metadata_columns(self) -> List[str]:
        """Return metadata columns to include in results.

        Returns:
            List[str]: ['company', 'doc_name', 'doc_type']
        """
        return ["company", "doc_name", "doc_type"]

    def get_dataset_name(self) -> str:
        """Return short name for FinanceBench.

        Returns:
            str: 'fb'
        """
        return "fb"
