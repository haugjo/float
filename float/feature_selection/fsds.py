"""Fast Feature Selection on Data Streams Method.

This module contains the Fast Feature Selection in Data Streams model that is able to select features via a sketching
algorithm without requiring supervision. The method was introduced by:
HUANG, Hao; YOO, Shinjae; KASIVISWANATHAN, Shiva Prasad. Unsupervised feature selection on data streams.
In: Proceedings of the 24th ACM International on Conference on Information and Knowledge Management. 2015. S. 1031-1040.

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
import numpy.linalg as ln
from numpy.typing import ArrayLike
from typing import Union, Optional

from float.feature_selection.base_feature_selector import BaseFeatureSelector


class FSDS(BaseFeatureSelector):
    """FSDS feature selector.

    This code is adopted from the official Python implementation of the authors with minor adaptations.
    """
    def __init__(self, n_total_features: int, n_selected_features: int, l: int = 0, m: Optional[int] = None,
                 B: Optional[Union[list, ArrayLike]] = None, k: int = 2, reset_after_drift: bool = False,
                 baseline: str = 'constant', ref_sample: Union[float, ArrayLike] = 0):
        """Inits the feature selector.

        Args:
            n_total_features: See description of base class.
            n_selected_features: See description of base class.
            l: Size of the matrix sketch with l << m.
            m: Size of the feature space (i.e. dimensionality).
            B: Matrix sketch.
            k: Number of singular vectors.
            reset_after_drift: See description of base class.
            baseline: See description of base class.
            ref_sample: See description of base class.
        """
        super().__init__(n_total_features=n_total_features, n_selected_features=n_selected_features,
                         supports_multi_class=True, reset_after_drift=reset_after_drift, baseline=baseline,
                         ref_sample=ref_sample)

        self._m_init = m
        self._B_init = B
        self._m = n_total_features if m is None else m
        self._B = [] if B is None else B
        self._l = l
        self._k = k

    def weight_features(self, X: ArrayLike, y: ArrayLike):
        """Updates feature weights.

        FSDS is an unsupervised approach and does not use target information.
        """
        Yt = X.T  # Algorithm assumes rows to represent features

        if self._l < 1:
            self._l = int(np.sqrt(self._m))

        if len(self._B) == 0:
            # For Y0, we need to first create an initial sketched matrix
            self._B = Yt[:, :self._l]
            C = np.hstack((self._B, Yt[:, self._l:]))
            n = Yt.shape[1] - self._l
        else:
            # Combine current sketched matrix with input at time t
            # C: m-by-(n+ell) matrix
            C = np.hstack((self._B, Yt))
            n = Yt.shape[1]

        U, s, V = ln.svd(C, full_matrices=False)
        U = U[:, :self._l]
        s = s[:self._l]
        V = V[:, :self._l]

        # Shrink step in Frequent Directions algorithm
        # (shrink singular values based on the squared smallest singular value)
        delta = s[-1] ** 2
        s = np.sqrt(s ** 2 - delta)

        # -- Extension of original code --
        # Replace nan values with 0 to prevent division by zero error for small batch numbers
        s = np.nan_to_num(s)

        # Update sketched matrix B
        # (focus on column singular vectors)
        self._B = np.dot(U, np.diag(s))

        # According to Section 5.1, for all experiments,
        # the authors set alpha = 2^3 * sigma_k based on the pre-experiment
        alpha = (2 ** 3) * s[self._k - 1]

        # Solve the ridge regression by using the top-k singular values
        # X: m-by-k matrix (k <= ell)
        D = np.diag(s[:self._k] / (s[:self._k] ** 2 + alpha))

        # -- Extension of original code --
        # Replace nan values with 0 to prevent division by zero error for small batch numbers
        D = np.nan_to_num(D)

        x = np.dot(U[:, :self._k], D)

        self.weights = np.amax(abs(x), axis=1)

    def reset(self):
        """Resets the feature selector."""
        self._m = self.n_total_features if self._m_init is None else self._m_init
        self._B = [] if self._B_init is None else self._B_init
