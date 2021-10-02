import numpy as np


def nogueira_stability(selected_features, n_total_features):
    """
    Returns the Nogueira measure for feature selection stability.

    todo: add reference to paper

    Args:
        selected_features (list): vector of selected features per time step
        n_total_features (int): total number of features

    Returns:
        float: the stability measure due to Nogueira et al.
    """
    Z = np.zeros([min(len(selected_features), 2), n_total_features])
    for row, col in enumerate(selected_features[-2:]):  # Todo: we compute stability between last two selections (do we need a larger window here?)
        Z[row, col] = 1

    try:
        M, d = Z.shape
        hatPF = np.mean(Z, axis=0)
        kbar = np.sum(hatPF)
        denom = (kbar / d) * (1 - kbar / d)
        stability = 1 - (M / (M - 1)) * np.mean(np.multiply(hatPF, 1 - hatPF)) / denom
    except ZeroDivisionError:
        stability = 0  # metric requires at least 2 measurements and thus runs an error at t=1

    return stability
