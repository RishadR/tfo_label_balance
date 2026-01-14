"""
Generate Training and Validation DataLoaders
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch


def get_holdout_dataloaders(
    dataset: pd.DataFrame,
    meta_data: Dict,
    device: torch.device,
    heldout_groups: List[str],
    batch_size: int = 32,
    shuffle: bool = True,
    include_synthetic_in_train: bool = False,
    include_synthetic_in_val: bool = False,
    **kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Generate training and validation DataLoaders from the dataset.

    :param dataset: The complete dataset as a pandas DataFrame. The dataset must contain these columns:
        - Feature columns as specified in meta_data["feature_names"]
        - Label column as specified in meta_data["label_name"]
        - Grouping column as specified in meta_data["grouping_column"]
        - "synthetic" column indicating whether a row is synthetic or real.
    :type dataset: pd.DataFrame
    :param meta_data: Metadata dictionary containing necessary information. The meta_data requires the following keys:
        - "feature_names": List of feature column names.
        - "label_name": The name of the label column.
        - "grouping_column": The name of the column used for grouping.
    :type meta_data: Dict
    :param device: The device to load the tensors onto.
    :type device: torch.device
    :param heldout_groups: List of group names to hold out for validation.
    :type heldout_groups: List[str]
    :param batch_size: Batch size for the DataLoaders.
    :type batch_size: int
    :param shuffle: Whether to shuffle the training data.
    :type shuffle: bool
    :param include_synthetic_in_train: Whether to include synthetic data in the training set.
    :type include_synthetic_in_train: bool
    :param include_synthetic_in_val: Whether to include synthetic data in the validation set.
    :type include_synthetic_in_val: bool
    :param kwargs: Additional keyword arguments passed directly to DataLoader.
    :type kwargs: dict
    :return: A tuple containing the training and validation DataLoaders.
    :rtype: Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]
    """
    grouping_column: str = meta_data["grouping_column"]
    feature_columns: List[str] = meta_data["feature_names"]
    label_name: str = meta_data["label_name"]
    grouping_column: str = meta_data["grouping_column"]

    # Split dataset into training and validation sets
    train_dataset = dataset[~dataset[grouping_column].isin(heldout_groups)]
    val_dataset = dataset[dataset[grouping_column].isin(heldout_groups)]

    # Handle inclusion/exclusion of synthetic data
    if not include_synthetic_in_train:
        train_dataset = train_dataset[train_dataset["synthetic"] == False]
    if not include_synthetic_in_val:
        val_dataset = val_dataset[val_dataset["synthetic"] == False]

    # Convert datasets to tensors
    def df_to_tensor_loader(df: pd.DataFrame) -> torch.utils.data.DataLoader:
        x = torch.tensor(df[feature_columns].to_numpy(), dtype=torch.float32).to(device)
        y = torch.tensor(df[label_name].to_numpy(), dtype=torch.float32).to(device).unsqueeze(1)
        tensor_dataset = torch.utils.data.TensorDataset(x, y)
        return torch.utils.data.DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

    train_loader = df_to_tensor_loader(train_dataset)
    val_loader = df_to_tensor_loader(val_dataset)

    return train_loader, val_loader
