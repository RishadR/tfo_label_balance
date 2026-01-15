"""
Implementation of various label balancing strategies for TFO datasets.
"""

from typing import List
import numpy as np
import pandas as pd
from tfo_label_balance.core import LabelBalancer
from tfo_label_balance.misc import compute_epsilon_ratio, compute_filtered_alpha, compute_epsilon, compute_filtered_beta
from typing import Dict


class LinRegLabelBalancer(LabelBalancer):
    """
    Group-wise Linear Regression using a physics-based constraint for generating balanced labels.

    Assumes that computed AC ratio equals to some patient-based constant times the epsilon ratio:
        AC1 / AC2 = alpha * (epsilon1 / epsilon2)   # AC1 should always be 735nm and AC2 should always be 850nm!
    where alpha is a constant for each patient/group.
    Uses the alpha to generate balance labels per group for each bin. Whichever bin has the highest count, the rest
    are synthetically balanced to match that count.

    meta_data should contain:
        - ac_ratio_names: List of column names corresponding to AC ratios (AC1 / AC2)
        - dc_names: List of column names corresponding to DC values
        - label_name: Name of the label column (saturation)
        - grouping_column: Name of the column to group by (e.g., patient ID)

    This creates a synthetic dataset with balanced labels and combines it with the original dataset. Additionally,
    we add a 'synthetic' boolean column to indicate whether a row is synthetic or original.
    """

    def __init__(
        self,
        bin_width: float = 0.05,
        value_range: tuple[float, float] = (0.1, 0.6),
        keep_threshold: float = 0.01,
        random_state: int = 42,
    ):
        required_keys = ["ac_ratio_names", "dc_names", "label_name", "grouping_column"]
        super().__init__(required_keys)
        self.bin_width = bin_width
        self.value_range = value_range
        self.keep_threshold = keep_threshold
        self.bins = np.arange(value_range[0], value_range[1] + bin_width, bin_width)
        self.random_state = random_state

    def balance(self, dataset: pd.DataFrame, meta_data: Dict) -> pd.DataFrame:
        # Set random seed for reproducibility
        np.random.seed(self.random_state)

        # Extract necessary metadata
        ac_ratio_names: List[str] = meta_data["ac_ratio_names"]
        dc_names: List[str] = meta_data["dc_names"]
        label_name: str = meta_data["label_name"]
        grouping_column: str = meta_data["grouping_column"]
        groups = dataset[grouping_column].unique().tolist()
        synthetic_data_ac_and_dc_and_sat = []
        synthetic_grouping_column = []

        # For each group, compute the alphas and balance the labels accordingly
        for group in groups:
            group_df = dataset[dataset[grouping_column] == group]
            saturation = group_df[label_name].to_numpy().flatten()

            # Compute how many datapoints to hit per bin
            per_bin_point_counts, _ = np.histogram(saturation, bins=self.bins)
            max_count = np.max(per_bin_point_counts)
            points_to_hit_per_bin = [max_count] * len(self.bins[:-1])
            points_to_generate = points_to_hit_per_bin - per_bin_point_counts

            dc_values = group_df[dc_names].to_numpy()
            dc_mean = np.array(np.mean(dc_values, axis=0))  # Shape: (num_dc_channels,)
            dc_mean = dc_mean.reshape(1, -1)  # Shape: (1, num_dc_channels)

            # Compute alphas for each AC ratio
            alphas = np.zeros(len(ac_ratio_names))
            for index, ac_ratio_name in enumerate(ac_ratio_names):
                ac_ratio = group_df[ac_ratio_name].to_numpy()
                alpha, _ = compute_filtered_alpha(ac_ratio, saturation, self.keep_threshold)
                # TODO: Determine if we keep these bad values or not
                alphas[index] = alpha

            # Generate synthetic points for each bin using pre-computed alphas
            ## All the following arrays are 2D where rows are data points
            synthetic_ac_ratios = np.zeros((np.sum(points_to_generate), len(ac_ratio_names)))
            synthetic_dcs = np.repeat(dc_mean, repeats=synthetic_ac_ratios.shape[0], axis=0)
            synthetic_saturations = np.zeros((synthetic_ac_ratios.shape[0], 1))
            current_pointer = 0
            for bin_index, points_to_generate_in_bin in enumerate(points_to_generate):
                # Ignore if no points to generate
                if points_to_generate_in_bin <= 0:
                    continue

                # Otherwise generate random saturation bins and create synthetic AC ratios
                bin_lower = self.bins[bin_index]
                bin_upper = self.bins[bin_index + 1]
                random_saturations = np.random.uniform(bin_lower, bin_upper, points_to_generate_in_bin)
                epsilon_ratios = compute_epsilon_ratio(random_saturations, 735.0, 850.0)  # THE ORDERING HAS TO MATCH
                temp_ac_ratios = epsilon_ratios.reshape(-1, 1) * alphas.reshape(1, -1)
                data_len = temp_ac_ratios.shape[0]
                synthetic_ac_ratios[current_pointer : current_pointer + data_len, :] = temp_ac_ratios
                synthetic_saturations[current_pointer : current_pointer + data_len, 0] = random_saturations
                current_pointer += data_len
            synthetic_data_ac_and_dc_and_sat.append(
                np.hstack((synthetic_ac_ratios, synthetic_dcs, synthetic_saturations))
            )
            synthetic_grouping_column.extend([group] * synthetic_ac_ratios.shape[0])

        # Combine all synthetic data and create a DataFrame
        synthetic_data_ac_and_dc_and_sat = np.vstack(synthetic_data_ac_and_dc_and_sat)
        synthetic_columns = ac_ratio_names + dc_names + [label_name]
        synthetic_df = pd.DataFrame(synthetic_data_ac_and_dc_and_sat, columns=synthetic_columns)
        synthetic_df["synthetic"] = True
        synthetic_df[grouping_column] = synthetic_grouping_column

        # Combine the original dataset with the synthetic dataset
        ## Keep only the required columns from the original dataset
        original_df = dataset[synthetic_columns + [grouping_column]].copy()
        original_df["synthetic"] = False

        # Combine the original dataset with the synthetic dataset
        balanced_df = pd.concat([original_df, synthetic_df], ignore_index=True)

        return balanced_df

    def __str__(self):
        return f"Linear Regression Label Balancer (bin_width={self.bin_width}, range={self.value_range})"


class NoLabelBalancer(LabelBalancer):
    """
    Emulate no label balancing. The returned dataset has the same format as a regular label balancer.
    """

    def __init__(self):
        required_keys = ["ac_ratio_names", "dc_names", "label_name", "grouping_column"]
        super().__init__(required_keys)

    def balance(self, dataset: pd.DataFrame, meta_data: Dict) -> pd.DataFrame:
        balanced_df = dataset.copy()
        ac_ratio_names: List[str] = meta_data["ac_ratio_names"]
        dc_names: List[str] = meta_data["dc_names"]
        label_name: str = meta_data["label_name"]
        grouping_column: str = meta_data["grouping_column"]
        columns_to_keep = ac_ratio_names + dc_names + [label_name, grouping_column]
        balanced_df = balanced_df[columns_to_keep]
        balanced_df["synthetic"] = False
        return balanced_df

    def __str__(self):
        return "No Label Balancer"


class LinRegACLabelBalancer(LabelBalancer):
    """
    Group-wise Linear Regression using a physics-based constraint for generating balanced labels.

    Assumes that each AC channel equals to some patient-based constant(\beta) times the extinction coefficient:
        AC = beta * epsilon   # where epsilon is computed using the saturation value at that point
    Uses the beta to generate balance labels per group for each bin. Whichever bin has the highest count, the rest
    are synthetically balanced to match that count.

    meta_data should contain:
        - ac_names: List of column names corresponding to AC values
        - dc_names: List of column names corresponding to DC values
        - label_name: Name of the label column (saturation)
        - grouping_column: Name of the column to group by (e.g., patient ID)

    This creates a synthetic dataset with balanced labels and combines it with the original dataset. Additionally,
    we add a 'synthetic' boolean column to indicate whether a row is synthetic or original.

    Extras:
        - Assumes two wavelengths: First half are for 735nm, second half are for 850nm
        - Uses compute_epsilon function from misc.py
        - The keep_threshold is used to filter out bad beta values when computing the mean beta per channel
        - For the synthethic DC values, uses mean of all DCs within that round + channel combo that are within
        3 standard deviations of the mean.
        - The saturation values should be in the range [0, 1]
    """

    def __init__(
        self,
        bin_width: float = 0.05,
        value_range: tuple[float, float] = (0.1, 0.6),
        keep_threshold: float = 0.01,
        random_state: int = 42,
    ):
        required_keys = ["ac_names", "dc_names", "label_name", "grouping_column"]
        super().__init__(required_keys)
        self.bin_width = bin_width
        self.value_range = value_range
        self.keep_threshold = keep_threshold
        self.bins = np.arange(value_range[0], value_range[1] + bin_width, bin_width)
        self.random_state = random_state

    def balance(self, dataset: pd.DataFrame, meta_data: Dict) -> pd.DataFrame:
        # Set random seed for reproducibility
        np.random.seed(self.random_state)

        # Extract necessary metadata
        ac_names: List[str] = meta_data["ac_names"]
        dc_names: List[str] = meta_data["dc_names"]
        label_name: str = meta_data["label_name"]
        grouping_column: str = meta_data["grouping_column"]
        groups = dataset[grouping_column].unique().tolist()
        synthetic_data_ac_and_dc_and_sat = []
        synthetic_grouping_column = []

        # For each group, compute the alphas and balance the labels accordingly
        for group in groups:
            group_df = dataset[dataset[grouping_column] == group]
            saturation = group_df[label_name].to_numpy().flatten()

            # Compute how many datapoints to hit per bin
            per_bin_point_counts, _ = np.histogram(saturation, bins=self.bins)
            max_count = np.max(per_bin_point_counts)
            points_to_hit_per_bin = [max_count] * len(self.bins[:-1])
            points_to_generate = points_to_hit_per_bin - per_bin_point_counts

            # Compute mean DC values for the group
            dc_values = group_df[dc_names].to_numpy()
            std_dev = np.std(dc_values, axis=0)
            unfiltered_mean = np.mean(dc_values, axis=0)
            ## Filter out DC values that are more than 3 std dev away from the mean
            filtered_dc_means = np.zeros((1, dc_values.shape[1])) # Shape: (1, num_dc_channels)
            for dc_index in range(dc_values.shape[1]):
                dc_column = dc_values[:, dc_index]
                indices_to_keep = np.where(
                    (dc_column >= unfiltered_mean[dc_index] - 3 * std_dev[dc_index])
                    & (dc_column <= unfiltered_mean[dc_index] + 3 * std_dev[dc_index])
                )[0]
                filtered_dc_means[0, dc_index] = np.mean(dc_column[indices_to_keep])

            # Compute betas for each AC 
            betas = np.zeros(len(ac_names))
            wavelengths = [735.0] * (len(ac_names) // 2) + [850.0] * (len(ac_names) // 2)
            for index, ac_name in enumerate(ac_names):
                ac = group_df[ac_name].to_numpy()
                beta, _ = compute_filtered_beta(ac, saturation, wavelengths[index], self.keep_threshold)
                # TODO: Determine if we keep these bad values or not
                betas[index] = beta

            # Generate synthetic points for each bin using pre-computed betas
            ## All the following arrays are 2D where rows are data points
            synthetic_acs = np.zeros((np.sum(points_to_generate), len(ac_names)))
            synthetic_dcs = np.repeat(filtered_dc_means, repeats=synthetic_acs.shape[0], axis=0)
            synthetic_saturations = np.zeros((synthetic_acs.shape[0], 1))
            current_pointer = 0
            for bin_index, points_to_generate_in_bin in enumerate(points_to_generate):
                # Ignore if no points to generate
                if points_to_generate_in_bin <= 0:
                    continue
                # Otherwise generate random saturation bins and create synthetic AC ratios
                bin_lower = self.bins[bin_index]
                bin_upper = self.bins[bin_index + 1]
                random_saturations = np.random.uniform(bin_lower, bin_upper, points_to_generate_in_bin)
                epsilon_values = np.zeros((points_to_generate_in_bin, len(ac_names)))   # Shape: (num_points, num_ac_channels)
                for ac_index in range(len(ac_names)):
                    wavelength = wavelengths[ac_index]
                    epsilon_values[:, ac_index] = compute_epsilon(random_saturations, wavelength)
                temp_acs = epsilon_values * betas.reshape(1, -1)    # Shape: (num_points, num_ac_channels)
                data_len = temp_acs.shape[0]
                ## Attach to main synthetic arrays
                synthetic_acs[current_pointer : current_pointer + data_len, :] = temp_acs
                synthetic_saturations[current_pointer : current_pointer + data_len, 0] = random_saturations
                current_pointer += data_len
            synthetic_data_ac_and_dc_and_sat.append(
                np.hstack((synthetic_acs, synthetic_dcs, synthetic_saturations))
            )
            synthetic_grouping_column.extend([group] * synthetic_acs.shape[0])

        # Combine all synthetic data and create a DataFrame
        synthetic_data_ac_and_dc_and_sat = np.vstack(synthetic_data_ac_and_dc_and_sat)
        synthetic_columns = ac_names + dc_names + [label_name]
        synthetic_df = pd.DataFrame(synthetic_data_ac_and_dc_and_sat, columns=synthetic_columns)
        synthetic_df["synthetic"] = True
        synthetic_df[grouping_column] = synthetic_grouping_column

        # Combine the original dataset with the synthetic dataset
        ## Keep only the required columns from the original dataset
        original_df = dataset[synthetic_columns + [grouping_column]].copy()
        original_df["synthetic"] = False

        # Combine the original dataset with the synthetic dataset
        balanced_df = pd.concat([original_df, synthetic_df], ignore_index=True)

        return balanced_df

    def __str__(self):
        return f"Linear Regression Label Balancer (bin_width={self.bin_width}, range={self.value_range})"
