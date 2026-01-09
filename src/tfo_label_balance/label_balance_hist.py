"""
Generate histogram plots for label balance analysis.
"""

from math import ceil
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from tfo_label_balance.label_balancers import LinRegLabelBalancer


def create_plot(
    original_dataset: pd.DataFrame,
    post_balance_dataset: pd.DataFrame,
    label_name: str,
    grouping_column: str,
    bin_width: float,
    bin_range: tuple[float, float],
) -> Figure:
    """
    Create one histogram plot per group in the dataset showing the label counts pre and post balancing.

    :param original_dataset: Original dataset before balancing.
    :type original_dataset: pd.DataFrame
    :param post_balance_dataset: Dataset after balancing.
    :type post_balance_dataset: pd.DataFrame
    :param label_name: Name of the label column.
    :type label_name: str
    :param grouping_column: Name of the column to group by.
    :type grouping_column: str
    :param bin_width: Width of the bins for the histogram.
    :type bin_width: float
    :param bin_range: Range of values for the bins.
    :type bin_range: tuple[float, float]
    :return: Matplotlib Figure object containing the plots.
    :rtype: Figure
    """
    groups = original_dataset[grouping_column].unique().tolist()
    num_groups = len(groups)
    row_count = ceil(num_groups / 2)
    fig, axes = plt.subplots(row_count, 2, figsize=(8, 4 * row_count), sharex=True)
    bins = np.arange(bin_range[0], bin_range[1] + bin_width, bin_width)
    axes = axes.flatten()
    has_useless_last_axis = num_groups % 2 == 1
    if has_useless_last_axis:
        fig.delaxes(axes[-1])
        axes = axes[:-1]

    for idx, group in enumerate(groups):
        plt.sca(axes[idx])
        original_group_data = original_dataset[
            original_dataset[grouping_column] == group
        ]
        balanced_group_data = post_balance_dataset[
            post_balance_dataset[grouping_column] == group
        ]

        # Plot histograms and get the counts
        counts_balanced, _, _ = axes[idx].hist(
            balanced_group_data[label_name],
            bins=bins,
            alpha=0.3,
            label="Balanced",
            color="orange",
        )
        counts_original, _, _ = axes[idx].hist(
            original_group_data[label_name],
            bins=bins,
            alpha=0.3,
            label="Original",
            color="blue",
        )

        # Calculate bin centers for the line plots
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Plot lines connecting the top of histogram bars
        axes[idx].plot(
            bin_centers,
            counts_original,
            color="blue",
            linewidth=2,
            marker="o",
            markersize=4,
        )
        axes[idx].plot(
            bin_centers,
            counts_balanced,
            color="orange",
            linewidth=2,
            marker="o",
            markersize=4,
        )

        axes[idx].set_title(f"Group: {group}")
        axes[idx].set_xlabel(label_name)
        axes[idx].set_ylabel("Count")
    axes[0].legend()
    return fig


if __name__ == "__main__":
    true_data = pd.read_csv("./data/combined_LLPSA2.csv")
    true_data["fSaO2"] /= 100.0  # Convert to fraction
    for idx in range(1, 6):
        true_data[f"AC_ratio_{idx}"] = (
            true_data[f"Amp_{idx}"].to_numpy() / true_data[f"Amp_{idx+5}"].to_numpy()
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
    balancer.check_meta_data(meta_data)
    balanced_data = balancer.balance(true_data, meta_data)
    print(f"Original data length: {len(true_data)}")
    print(f"Balanced data length: {len(balanced_data)}")
    print(f"New points added: {len(balanced_data) - len(true_data)}")

    # Plotting
    fig = create_plot(
        original_dataset=true_data,
        post_balance_dataset=balanced_data,
        label_name="fSaO2",
        grouping_column="experiment_id",
        bin_width=bin_width,
        bin_range=value_range,
    )
    fig.tight_layout()
    fig.savefig("./figures/label_balance_histogram.png")
    print("Histogram saved to ./figures/label_balance_histogram.png")
