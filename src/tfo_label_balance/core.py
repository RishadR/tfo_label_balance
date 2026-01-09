"""
Core definitions for the tfo_label_balance package.
"""
from abc import ABC, abstractmethod
from typing import Dict
import pandas as pd

class LabelBalancer(ABC):
    """
    Abstract base class for label balancing strategies in TFO datasets.
    """
    def __init__(self, required_keys: list[str]) -> None:
        self.meta_data_required_keys = required_keys

    @abstractmethod
    def balance(self, dataset: pd.DataFrame, meta_data: Dict) -> pd.DataFrame:
        """
        Balance the labels in the given dataset.

        Args:
            dataset: The dataset to be balanced.
            meta_data: The metadata dictionary containing necessary information.
        Returns:
            A new dataset with balanced labels.
        """
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        pass
    
    def check_meta_data(self, meta_data: Dict) -> None:
        """
        Check if the required keys are present in the meta_data dictionary.

        Args:
            meta_data: The metadata dictionary to check.
        Raises:
            KeyError: If any required key is missing.
        """
        for key in self.meta_data_required_keys:
            if key not in meta_data:
                raise KeyError(f"Missing required meta_data key: {key}")
    
    