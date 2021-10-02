from float.feature_selection.base_feature_selector import BaseFeatureSelector
import numpy as np
import math


class OFS(BaseFeatureSelector):
    """
    Online Feature Selection.

    Based on a paper by Wang et al. 2014. Feature Selection for binary classification.
    This code is an adaptation of the official Matlab implementation.
    """
    def __init__(self, n_total_features, n_selected_features, reset_after_drift=False, baseline='constant', ref_sample=0):
        """
        Initializes the OFS feature selector.

        Args:
            n_total_features (int): total number of features
            n_selected_features (int): number of selected features
            reset_after_drift (bool): indicates whether to reset the predictor after a drift was detected
            baseline (str): identifier of baseline method (value to replace non-selected features with)
            ref_sample (float | np.ndarray): integer (in case of 'constant' baseline) or sample used to obtain the baseline
        """
        super().__init__(n_total_features, n_selected_features, supports_multi_class=False,
                         reset_after_drift=reset_after_drift, baseline=baseline, ref_sample=ref_sample)

    def weight_features(self, X, y):
        """
        Given a batch of observations and corresponding labels, computes feature weights.

        Args:
            X (np.ndarray): samples of current batch
            y (np.ndarray): labels of current batch
        """
        eta = 0.2
        lamb = 0.01

        for x_b, y_b in zip(X, y):  # perform feature selection for each instance in batch
            # Convert label to -1 and 1
            y_b = -1 if y_b == 0 else 1

            f = np.dot(self.raw_weight_vector, x_b)  # prediction

            if y_b * f <= 1:  # update classifier w
                self.raw_weight_vector = self.raw_weight_vector + eta * y_b * x_b
                self.raw_weight_vector = self.raw_weight_vector * min(1, 1 / (math.sqrt(lamb) * np.linalg.norm(self.raw_weight_vector)))

    def reset(self):
        """
        Reset weight vector
        """
        self.raw_weight_vector = np.zeros(self.n_total_features)
