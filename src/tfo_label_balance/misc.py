"""
Miscellaneous utility functions for TFO label balancing.
"""

from tfo_sim2.four_layer_model_optical_props_table import get_blood_filled_tissue_mu_a
import numpy as np


def compute_epsilon(saturation: np.ndarray, wavelength: float) -> np.ndarray:
    """
    Compute the epsilon value for given saturation at a specific wavelength.

    Args:
        saturation: 1D Array of saturation values (0 to 1).
        wavelength: Wavelength in nm.
    Returns:
        Array of epsilon values as a 1D numpy array.
    """
    saturation = saturation.flatten()
    epsilon = np.array([get_blood_filled_tissue_mu_a(1.0, 1.0, x, wavelength, 1.0, 0.0) for x in saturation])
    return epsilon


def compute_epsilon_ratio(
    saturation_array: np.ndarray, wavelength1: float = 735.0, wavelength2: float = 850.0
) -> np.ndarray:
    """
    Compute the epsilon ratio for given saturation values at two wavelengths.

    Args:
        saturation_array: 1D Array of saturation values (0 to 1).
        wavelength1: First wavelength in nm.
        wavelength2: Second wavelength in nm.
    Returns:
        Array of epsilon ratios - epsilon(wavelength1) / epsilon(wavelength2) as a 1D numpy array.
    """
    saturation_array = saturation_array.flatten()
    epsilon1 = compute_epsilon(saturation_array, wavelength1)
    epsilon2 = compute_epsilon(saturation_array, wavelength2)
    epsilon_ratio = epsilon1 / epsilon2
    return epsilon_ratio


def compute_filtered_alpha(
    ac_ratio: np.ndarray, saturation_array: np.ndarray, keep_threshold: float = 0.1
) -> tuple[float, list[int]]:
    """
    Compute the filtered alpha value from AC ratio and saturation data.
    Uses the fitting equation:
        AC1 / AC2 = alpha * (epsilon1 / epsilon2)
    and filters out outliers based on the keep_threshold.

    Args:
        ac_ratio: 1D Array of AC ratios (AC1 / AC2). Always assumes AC1 is 735.0nm and AC2 is 850.0nm.
        saturation_array: 1D Array of saturation values (0 to 1).
        keep_threshold: Threshold for filtering outliers based on error.
    Returns:
        A tuple containing:
            - The computed alpha value after filtering.
            - List of indices that were kept after filtering.
    """
    ac_ratio = ac_ratio.flatten()
    saturation_array = saturation_array.flatten()
    epsilon_ratio = compute_epsilon_ratio(saturation_array)
    ac_ratio = ac_ratio.reshape(-1, 1)
    epsilon_ratio = epsilon_ratio.reshape(-1, 1)
    alpha, _, __, ___ = np.linalg.lstsq(epsilon_ratio, ac_ratio, rcond=None)
    error_per_row = np.square(ac_ratio - epsilon_ratio * alpha)
    indices_to_keep = np.where(error_per_row.flatten() < keep_threshold)[0]
    ac_ratio_filtered = ac_ratio[indices_to_keep]
    epsilon_ratio_filtered = epsilon_ratio[indices_to_keep]
    alpha_filtered, _, __, ___ = np.linalg.lstsq(epsilon_ratio_filtered, ac_ratio_filtered, rcond=None)
    alpha_filtered = float(alpha_filtered[0][0])
    return alpha_filtered, indices_to_keep.tolist()


def compute_filtered_beta(
    ac_channel: np.ndarray, saturation_array: np.ndarray, wavelength: float, keep_threshold: float = 0.1
) -> tuple[float, list[int]]:
    """
    Compute the filtered beta value from AC channel and saturation data.
    Uses the fitting equation:
        AC = beta * epsilon
    and filters out outliers based on the keep_threshold.

    Args:
        ac_channel: 1D Array of AC channel values.
        saturation_array: 1D Array of saturation values (0 to 1).
        wavelength: Wavelength in nm.
        keep_threshold: Threshold for filtering outliers based on error.
    Returns:
        A tuple containing:
            - The computed beta value after filtering.
            - List of indices that were kept after filtering.
    """
    ac_channel = ac_channel.flatten()
    saturation_array = saturation_array.flatten()
    epsilon = compute_epsilon(saturation_array, wavelength)
    ac_channel = ac_channel.reshape(-1, 1)
    epsilon = epsilon.reshape(-1, 1)
    beta, _, __, ___ = np.linalg.lstsq(epsilon, ac_channel, rcond=None)
    error_per_row = np.square(ac_channel - epsilon * beta)
    indices_to_keep = np.where(error_per_row.flatten() < keep_threshold)[0]
    ac_channel_filtered = ac_channel[indices_to_keep]
    epsilon_filtered = epsilon[indices_to_keep]
    beta_filtered, _, __, ___ = np.linalg.lstsq(epsilon_filtered, ac_channel_filtered, rcond=None)
    beta_filtered = float(beta_filtered[0][0])
    return beta_filtered, indices_to_keep.tolist()
