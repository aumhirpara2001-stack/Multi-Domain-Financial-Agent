"""Base dataset adapter abstract class."""

from abc import ABC, abstractmethod
from typing import Optional, List
import pandas as pd


class BaseDatasetAdapter(ABC):
    """Abstract base class for dataset adapters.

    Any dataset used for bulk testing must implement this interface.
    This ensures consistent access to questions, answers, and metadata
    across different datasets.
    """

    @abstractmethod
    def load_dataset(self) -> pd.DataFrame:
        """Load the dataset and return as a pandas DataFrame.

        Returns:
            pd.DataFrame: Dataset with all required columns
        """
        pass

    @abstractmethod
    def get_question_column(self) -> str:
        """Return the name of the column containing questions.

        Returns:
            str: Column name for questions
        """
        pass

    @abstractmethod
    def get_answer_column(self) -> str:
        """Return the name of the column containing gold answers.

        Returns:
            str: Column name for gold answers
        """
        pass

    @abstractmethod
    def get_question_type_column(self) -> Optional[str]:
        """Return the name of the column containing question types.

        Returns:
            Optional[str]: Column name for question types, or None if not available
        """
        pass

    @abstractmethod
    def get_metadata_columns(self) -> List[str]:
        """Return list of additional metadata columns to include in results.

        Returns:
            List[str]: List of metadata column names (e.g., company, doc_name)
        """
        pass

    @abstractmethod
    def get_dataset_name(self) -> str:
        """Return a short name/abbreviation for this dataset.

        This will be used in output filenames.

        Returns:
            str: Short dataset name (e.g., 'fb' for FinanceBench)
        """
        pass
