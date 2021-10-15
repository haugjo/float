"""Extremal Feature Selection Method.

This module contains the Extremal Feature Selection model introduced by:
CARVALHO, Vitor R.; COHEN, William W. Single-pass online learning: Performance, voting schemes and online feature
selection. In: Proceedings of the 12th ACM SIGKDD international conference on Knowledge discovery and data mining.
2006. S. 548-553.

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
from sklearn.preprocessing import MinMaxScaler
from typing import Optional, Union

from float.feature_selection.base_feature_selector import BaseFeatureSelector


class EFS(BaseFeatureSelector):
    """Extremal feature selector.

    This feature selection algorithm uses the weights of a Modified Balanced Winnow classifier.
    """
    def __init__(self, n_total_features: int, n_selected_features: int, u: Optional[ArrayLike] = None,
                 v: Optional[ArrayLike] = None, theta: float = 1, M: float = 1, alpha: float = 1.5, beta: float = 0.5,
                 reset_after_drift: bool = False, baseline: str = 'constant', ref_sample: Union[float, ArrayLike] = 0):
        """Initializes the feature selector.

        Args:
            n_total_features: See description of base class.
            n_selected_features: See description of base class.
            u: Initial positive model weights of the Winnow algorithm.
            v: Initial negative model weights of the Winnow algorithm.
            theta: Threshold parameter.
            M (float): Margin parameter.
            alpha (float): Promotion parameter.
            beta (float): Demotion parameter.
            reset_after_drift: See description of base class.
            baseline: See description of base class.
            ref_sample: See description of base class.
        """
        super().__init__(n_total_features=n_total_features, n_selected_features=n_selected_features,
                         supports_multi_class=False, reset_after_drift=reset_after_drift, baseline=baseline,
                         ref_sample=ref_sample)

        self._u_init = u
        self._v_init = v
        self._u = np.ones(n_total_features) * 2 if u is None else u
        self._v = np.ones(n_total_features) if v is None else v

        self._theta = theta
        self._M = M
        self._alpha = alpha
        self._beta = beta

    def weight_features(self, x: ArrayLike, y: ArrayLike):
        """Updates feature weights."""
        # Iterate over all elements in the batch
        for x_b, y_b in zip(x, y):

            # Convert label to -1 and 1
            y_b = -1 if y_b == 0 else 1

            # Note, the original algorithm here adds a "bias" feature that is always 1

            # Normalize x_b
            x_b = MinMaxScaler().fit_transform(x_b.reshape(-1, 1)).flatten()

            # Calculate score
            score = np.dot(x_b, self._u) - np.dot(x_b, self._v) - self._theta

            # If prediction was mistaken
            if score * y_b <= self._M:
                # Update models for all features j
                for j, _ in enumerate(self._u):
                    if y_b > 0:
                        self._u[j] = self._u[j] * self._alpha * (1 + x_b[j])
                        self._v[j] = self._v[j] * self._beta * (1 - x_b[j])
                    else:
                        self._u[j] = self._u[j] * self._beta * (1 - x_b[j])
                        self._v[j] = self._v[j] * self._alpha * (1 + x_b[j])

        # Compute importance score of features
        self.weights = abs(self._u - self._v)

    def reset(self):
        """Resets the feature selector."""
        self._u = np.ones(self.n_total_features) * 2 if self._u_init is None else self._u_init
        self._v = np.ones(self.n_total_features) if self._v_init is None else self._v_init
