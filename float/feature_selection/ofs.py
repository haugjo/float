"""Online Feature Selection Method.

This module contains the Online Feature Selection model based on a Perceptron, which was introduced by:
WANG, Jialei, et al. Online feature selection and its applications. IEEE Transactions on knowledge and data engineering,
2013, 26. Jg., Nr. 3, S. 698-710.

Copyright (C) 2021 Johannes Haug

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
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
            n_total_features: See description of base class.
            n_selected_features: See description of base class.
            reset_after_drift: See description of base class.
            baseline: See description of base class.
            ref_sample: See description of base class.
        """
        super().__init__(n_total_features=n_total_features,
                         n_selected_features=n_selected_features,
                         supports_multi_class=False,
                         reset_after_drift=reset_after_drift,
                         baseline=baseline,
                         ref_sample=ref_sample)

    def weight_features(self, X: ArrayLike, y: ArrayLike):
        """Updates feature weights."""
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
