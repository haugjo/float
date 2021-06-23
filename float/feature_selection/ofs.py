from float.feature_selection.feature_selector import FeatureSelector
import numpy as np
import math


class OFS(FeatureSelector):
    """
    Online Feature Selection.

    Based on a paper by Wang et al. 2014. Feature Selection for binary classification.
    This code is an adaptation of the official Matlab implementation.
    """
    def __init__(self, n_total_features, n_selected_features, nogueira_window_size=None):
        super().__init__(n_total_features, n_selected_features, supports_multi_class=False,
                         supports_streaming_features=False, nogueira_window_size=nogueira_window_size)

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
