"""Online Feature Selection Method.

This module contains the Online Feature Selection model based on a Perceptron, which was introduced by:
WANG, Jialei, et al. Online feature selection and its applications. IEEE Transactions on knowledge and data engineering,
2013, 26. Jg., Nr. 3, S. 698-710.

Copyright (C) 2022 Johannes Haug.
"""
import numpy as np
from numpy.typing import ArrayLike
import math
from typing import Union

from float.feature_selection.base_feature_selector import BaseFeatureSelector


class OFS(BaseFeatureSelector):
    """OFS feature selector.

    This feature selector uses the weights of a Perceptron classifier.
    """
    def __init__(self,
                 n_total_features: int,
                 n_selected_features: int,
                 reset_after_drift: bool = False,
                 baseline: str = 'constant',
                 ref_sample: Union[float, ArrayLike] = 0):
        """Inits the feature selector.

        Args:
            n_total_features: The total number of features.
            n_selected_features: The number of selected features.
            reset_after_drift: A boolean indicating if the change detector will be reset after a drift was detected.
            baseline:
                A string identifier of the baseline method. The baseline is the value that we substitute non-selected
                features with. This is necessary, because most online learning models are not able to handle arbitrary
                patterns of missing data.
            ref_sample:
                A sample used to compute the baseline. If the constant baseline is used, one needs to provide a single
                float value.
        """
        super().__init__(n_total_features=n_total_features,
                         n_selected_features=n_selected_features,
                         supports_multi_class=False,
                         reset_after_drift=reset_after_drift,
                         baseline=baseline,
                         ref_sample=ref_sample)

    def weight_features(self, X: ArrayLike, y: ArrayLike):
        """Updates feature weights.

        Args:
            X: Array/matrix of observations.
            y: Array of corresponding labels.
        """
        eta = 0.2  # Default parameters as proposed by the authors
        lamb = 0.01

        for x_b, y_b in zip(X, y):  # Perform feature selection for each instance in batch
            # Convert label to -1 and 1
            y_b = -1 if y_b == 0 else 1

            f = np.dot(self.weights, x_b)  # Prediction

            if y_b * f <= 1:  # Update classifier weights
                self.weights = self.weights + eta * y_b * x_b
                self.weights = self.weights * min(1, 1 / (math.sqrt(lamb) * np.linalg.norm(self.weights)))

    def reset(self):
        """Resets the feature selector."""
        self.weights = np.zeros(self.n_total_features)
