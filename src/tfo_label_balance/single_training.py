"""
Code to run a single instance of model training
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from typing import Dict, Callable, Optional, List
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, RobustScaler
import numpy as np


def train_model(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    evaluator_fn: Callable[[nn.Module, DataLoader, torch.device], Dict[str, float]],
    num_epochs: int,
    scheduler: Optional[Callable[[torch.optim.Optimizer, int, Dict[str, float]], torch.optim.Optimizer]] = None,
) -> Dict[str, List[float]]:
    """
    Train a model

    Args:
        model: PyTorch model to train
        device: Torch device (cuda/cpu)
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer for training
        loss_fn: Loss function for training
        evaluator_fn: Function that takes (model, dataloader, device) and returns
                     a dict of metric names to float values
        num_epochs: Number of epochs to train
        scheduler: Optional learning rate scheduler

    Returns:
        Dictionary containing training history with keys:
            - 'train_loss': List of training losses per epoch
            - 'train_metrics': List of dicts containing training metrics per epoch
            - 'val_metrics': List of dicts containing validation metrics per epoch
    """

    # Initialize history storage
    history = {"train_loss": [], "train_metrics": [], "val_metrics": []}

    # Move model to device
    model.to(device)

    print(f"\n{'='*80}")
    print(f"Starting Training for {num_epochs} epochs")
    print(f"Device: {device}")
    print(f"{'='*80}\n")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for batch_idx, (features, labels) in enumerate(train_pbar):
            # Move data to device
            features = features.to(device)
            labels = labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass with actual features
            predictions = model(features)

            # Compute label loss
            label_loss = loss_fn(predictions, labels)
            total_loss = label_loss  # Keeping this here so I can add more losses later if needed

            # Backward pass
            total_loss.backward()

            # Optimizer step
            optimizer.step()

            # Accumulate losses
            epoch_loss += total_loss.item()
            num_batches += 1

            # Update progress bar
            train_pbar.set_postfix(
                {
                    "loss": f"{total_loss.item():.3E}",
                }
            )

        # Average losses for the epoch
        avg_train_loss = epoch_loss / num_batches

        # Store training losses
        history["train_loss"].append(avg_train_loss)

        # Evaluation phase
        print(f"\nEvaluating on training set...")
        train_metrics = evaluator_fn(model, train_loader, device)
        history["train_metrics"].append(train_metrics)

        print(f"Evaluating on validation set...")
        val_metrics = evaluator_fn(model, val_loader, device)
        history["val_metrics"].append(val_metrics)

        # Pretty print results
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{num_epochs} Summary:")
        print(f"{'-'*80}")
        print(f"Training Loss:    {avg_train_loss:.4E}")
        print(f"\nTraining Metrics:")
        for metric_name, metric_value in train_metrics.items():
            print(f"  - {metric_name}: {metric_value:.4E}")
        print(f"\nValidation Metrics:")
        for metric_name, metric_value in val_metrics.items():
            print(f"  - {metric_name}: {metric_value:.4E}")
        print(f"{'='*80}\n")

        # Update scheduler if provided
        if scheduler is not None:
            optimizer = scheduler(optimizer, epoch, val_metrics)

    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"{'='*80}\n")
    return history


def unscaled_mae_evaluator(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    y_scaler: StandardScaler | RobustScaler | None = None,
):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)
            predictions = model(features)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(predictions.cpu().numpy())
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    if y_scaler is not None:
        all_labels = y_scaler.inverse_transform(all_labels)
        all_preds = y_scaler.inverse_transform(all_preds)
    mae = np.mean(np.abs(all_labels - all_preds))
    return {"MAE": float(mae)}


def randalls_evaluator(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    y_scaler: StandardScaler | RobustScaler | None = None,
):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)
            predictions = model(features)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(predictions.cpu().numpy())
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    if y_scaler is not None:
        all_labels = y_scaler.inverse_transform(all_labels)
        all_preds = y_scaler.inverse_transform(all_preds)

    mae = np.mean(np.abs(all_labels - all_preds))
    correlation = np.corrcoef(all_labels.flatten(), all_preds.flatten())[0, 1]
    coeffs = np.polyfit(all_labels.flatten(), all_preds.flatten(), 1)
    slope = coeffs[0]
    intercept = coeffs[1]
    return {
        "MAE": float(mae),
        "Correlation": float(correlation),
        "Linear_Fit_Slope": float(slope),
        "Linear_Fit_Intercept": float(intercept),
    }
