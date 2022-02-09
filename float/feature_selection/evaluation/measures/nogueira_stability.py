"""Nogueira Feature Set Stability Measure.

This function returns the feature set stability measure introduced by:
NOGUEIRA, Sarah; SECHIDIS, Konstantinos; BROWN, Gavin. On the stability of feature selection algorithms.
J. Mach. Learn. Res., 2017, 18. Jg., Nr. 1, S. 6345-6398.

Copyright (C) 2022 Johannes Haug.
"""
import numpy as np
from typing import List


def nogueira_stability(selected_features_history: List[list], n_total_features: int) -> float:
    """Calculates the Nogueira measure for feature selection stability.

    Args:
        selected_features_history: A list of all selected feature vectors obtained over time.
        n_total_features: The total number of features.

    Returns:
        float: The feature set stability due to Nogueira et al.
    """
    Z = np.zeros([min(len(selected_features_history), 2), n_total_features])
    for row, col in enumerate(selected_features_history[-2:]):  # Compare most currently selected feature sets.
        Z[row, col] = 1

    try:
        M, d = Z.shape
        hatPF = np.mean(Z, axis=0)
        kbar = np.sum(hatPF)
        denom = (kbar / d) * (1 - kbar / d)
        stability = 1 - (M / (M - 1)) * np.mean(np.multiply(hatPF, 1 - hatPF)) / denom
    except ZeroDivisionError:
        stability = 0  # The measure requires at least 2 measurements and thus runs an error at t=1

    return stability
