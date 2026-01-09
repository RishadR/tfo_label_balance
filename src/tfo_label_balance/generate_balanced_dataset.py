"""
Run this to generate balanced datasets and store them along with metadata.
"""
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from tfo_label_balance.label_balancers import LinRegLabelBalancer, NoLabelBalancer


if __name__ == "__main__":
    og_data = pd.read_csv("./data/combined_LLPSA2.csv")
    og_data["fSaO2"] /= 100.0  # Convert to fraction
    for idx in range(1, 6):
        og_data[f"AC_ratio_{idx}"] = (
            og_data[f"Amp_{idx}"].to_numpy() / og_data[f"Amp_{idx+5}"].to_numpy()
        )
    meta_data = {
        "ac_ratio_names": [f"AC_ratio_{i}" for i in range(1, 6)],
        "dc_names": [f"DC_{i}" for i in range(1, 11)],
        "label_name": "fSaO2",
        "grouping_column": "experiment_id",
    }
    bin_width = 0.05
    value_range = (0.1, 0.6)
    balancer = LinRegLabelBalancer(bin_width=bin_width, value_range=value_range)
    # balancer = NoLabelBalancer()
    balancer.check_meta_data(meta_data)
    balanced_data = balancer.balance(og_data, meta_data)
    print(f"Original data length: {len(og_data)}")
    print(f"Balanced data length: {len(balanced_data)}")
    print(f"New points added: {len(balanced_data) - len(og_data)}")
    
    balanced_data.to_csv("./data/balanced_dataset.csv", index=False)
    with open("./data/balanced_dataset.yaml", "w+") as f:
        yaml.dump(meta_data, f)
    