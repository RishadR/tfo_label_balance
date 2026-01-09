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
    **kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Generate training and validation DataLoaders from the dataset.

    :param dataset: The complete dataset as a pandas DataFrame.
    :type dataset: pd.DataFrame
    :param meta_data: Metadata dictionary containing necessary information. The meta_data requires the following keys:
        - "ac_ratio_names": List of column names for AC ratios.
        - "dc_names": List of column names for DC values.
        - "label_name": The name of the label column.
        - "grouping_column": The name of the column used for grouping.
    Out of these, the ac_ratio_names and dc_names are set as features and the label_name is set as the target variable.
    :type meta_data: Dict
    :param device: The device to load the tensors onto.
    :type device: torch.device
    :param heldout_groups: List of group names to hold out for validation.
    :type heldout_groups: List[str]
    :param batch_size: Batch size for the DataLoaders.
    :type batch_size: int
    :param kwargs: Additional keyword arguments passed directly to DataLoader.
    :return: A tuple containing the training and validation DataLoaders.
    :rtype: Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]
    """
    grouping_column: str = meta_data["grouping_column"]
    ac_ratio_names: List[str] = meta_data["ac_ratio_names"]
    dc_names: List[str] = meta_data["dc_names"]
    label_name: str = meta_data["label_name"]
    grouping_column: str = meta_data["grouping_column"]
    feature_columns = ac_ratio_names + dc_names

    # Split dataset into training and validation sets
    val_dataset = dataset[dataset[grouping_column].isin(heldout_groups)]
    train_dataset = dataset[~dataset[grouping_column].isin(heldout_groups)]

    # Convert datasets to tensors
    def df_to_tensor_loader(df: pd.DataFrame) -> torch.utils.data.DataLoader:
        x = torch.tensor(df[feature_columns].to_numpy(), dtype=torch.float32).to(device)
        y = torch.tensor(df[label_name].to_numpy(), dtype=torch.float32).to(device).unsqueeze(1)
        tensor_dataset = torch.utils.data.TensorDataset(x, y)
        return torch.utils.data.DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

    train_loader = df_to_tensor_loader(train_dataset)
    val_loader = df_to_tensor_loader(val_dataset)

    return train_loader, val_loader
