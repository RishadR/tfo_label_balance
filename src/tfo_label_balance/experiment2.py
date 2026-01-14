"""
Run experiment 1: Holdout group evaluation
"""

import yaml
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from tfo_label_balance.datagen import get_holdout_dataloaders
from tfo_label_balance.single_training import train_model, unscaled_mae_evaluator, randalls_evaluator


data = pd.read_csv("./data/balanced_dataset.csv")
with open("./data/balanced_dataset.yaml", "r") as f:
    meta_data = yaml.safe_load(f)
all_groups = data[meta_data["grouping_column"]].unique().tolist()
heldout_group = [all_groups[1]]  # Hold out the first group for validation
feature_names = meta_data["ac_ratio_names"] + meta_data["dc_names"]
label_name = meta_data["label_name"]    # Sat values are in the range [0, 1]
feature_len = len(feature_names)

# Scaling
feature_scaler = StandardScaler()
label_scaler = StandardScaler()
data[feature_names] = feature_scaler.fit_transform(data[feature_names])
data[[label_name]] = label_scaler.fit_transform(data[[label_name]])

# device = torch.device("cpu")
device = torch.device("cuda")
meta_data["feature_names"] = feature_names  # Add feature names to meta_data before passing onto dataloader
train_loader, val_loader = get_holdout_dataloaders(
    data,
    meta_data,
    device=device,
    heldout_groups=heldout_group,
    batch_size=32,
    shuffle=True,
    include_synthetic_in_train=True,
    include_synthetic_in_val=False,
)

model = torch.nn.Sequential(
    torch.nn.Linear(feature_len, 32),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(32),
    torch.nn.Linear(32, 8),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(8),
    torch.nn.Linear(8, 1),
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
loss_fn = torch.nn.MSELoss()
# evaluator = lambda model, data_loader, device: unscaled_mae_evaluator(model, data_loader, device, y_scaler=label_scaler)
evaluator = lambda model, data_loader, device: randalls_evaluator(model, data_loader, device, y_scaler=label_scaler)
train_log = train_model(model, device, train_loader, val_loader, optimizer, loss_fn, evaluator, num_epochs=50)
print("Experiment 1 complete.")
