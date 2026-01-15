"""
Run experiment 1: Holdout group evaluation
"""

import yaml
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from tfo_label_balance.datagen import get_holdout_dataloaders
from tfo_label_balance.single_training import train_model, unscaled_mae_evaluator, randalls_evaluator


data = pd.read_csv("./data/combined_LLPSA2.csv")
# with open("./data/balanced_dataset.yaml", "r") as f:
#     meta_data = yaml.safe_load(f)
for idx in range(10):
    data[f"EPR_{idx + 1}"] = data.iloc[:, idx].to_numpy() / data.iloc[:, idx + 10].to_numpy()
data["synthetic"] = False
data["fSaO2"] /= 100.0  # Scale to [0, 1]

all_groups = data["experiment_id"].unique().tolist()
heldout_group = [all_groups[1]]  # Hold out the first group for validation
feature_names = [f"EPR_{idx + 1}" for idx in range(10)]
label_name = "fSaO2"
feature_len = len(feature_names)

# Scaling
feature_scaler = StandardScaler()
label_scaler = StandardScaler()
data[feature_names] = feature_scaler.fit_transform(data[feature_names])
data[[label_name]] = label_scaler.fit_transform(data[[label_name]])

# device = torch.device("cpu")
device = torch.device("cuda")
meta_data = {
    "feature_names": feature_names,
    "label_name": label_name,
    "grouping_column": "experiment_id"
}
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
    torch.nn.Linear(feature_len, 12),
    torch.nn.ReLU(),
    torch.nn.Linear(12, 6),
    torch.nn.BatchNorm1d(6),
    torch.nn.ReLU(),
    torch.nn.Linear(6, 1),
).to(device)

# Apply Kaiming initialization to linear layers
for layer in model.modules():
    if isinstance(layer, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(layer.weight.data, nonlinearity='relu')

optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)
loss_fn = torch.nn.MSELoss()
# evaluator = lambda model, data_loader, device: unscaled_mae_evaluator(model, data_loader, device, y_scaler=label_scaler)
evaluator = lambda model, data_loader, device: randalls_evaluator(model, data_loader, device, y_scaler=label_scaler)
train_log = train_model(model, device, train_loader, val_loader, optimizer, loss_fn, evaluator, num_epochs=50)
print("Experiment 2 complete.")
