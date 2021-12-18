"""Base Online Feature Selection Module.

This module encapsulates functionality for online feature weighting and selection.
The abstract BaseFeatureSelector class should be used as super class for all online feature selection methods.

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
from abc import ABCMeta, abstractmethod
import numpy as np
from numpy.typing import ArrayLike
from numpy.random import Generator
from typing import Union, List
import warnings


class BaseFeatureSelector(metaclass=ABCMeta):
    """Abstract base class for online feature selection methods.

    Attributes:
        n_total_features (int): The total number of features.
        n_selected_features (int): The number of selected features.
        supports_multi_class (bool):
            True if the feature selection model supports multi-class classification, False otherwise.
        reset_after_drift (bool):
            A boolean indicating if the change detector will be reset after a drift was detected.
        baseline (str):
            A string identifier of the baseline method. The baseline is the value that we substitute non-selected
            features with. This is necessary, because most online learning models are not able to handle arbitrary
            patterns of missing data.
        ref_sample (ArrayLike | float):
            A sample used to compute the baseline. If the constant baseline is used, one needs to provide a single
            float value.
        weights (ArrayLike): The current (raw) feature weights.
        selected_features (ArrayLike): The indices of all currently selected features.
        weights_history (List[list]): A list of all absolute feature weight vectors obtained over time.
        selected_features_history (List[list]): A list of all selected feature vectors obtained over time.
    """

    def __init__(self,
                 n_total_features: int,
                 n_selected_features: int,
                 supports_multi_class: bool,
                 reset_after_drift: bool,
                 baseline: str,
                 ref_sample: Union[float, ArrayLike]):
        """Inits the feature selector.

        Args:
            n_total_features: The total number of features.
            n_selected_features: The number of selected features.
            supports_multi_class:
                True if the feature selection model supports multi-class classification, False otherwise.
            reset_after_drift: A boolean indicating if the change detector will be reset after a drift was detected.
            baseline:
                A string identifier of the baseline method. The baseline is the value that we substitute non-selected
                features with. This is necessary, because most online learning models are not able to handle arbitrary
                patterns of missing data.
            ref_sample:
                A sample used to compute the baseline. If the constant baseline is used, one needs to provide a single
                float value.
        """
        self.n_total_features = n_total_features
        self.n_selected_features = n_selected_features
        self.supports_multi_class = supports_multi_class
        self.reset_after_drift = reset_after_drift
        self.baseline = baseline
        self.ref_sample = ref_sample

        self.weights = np.zeros(self.n_total_features)
        self.selected_features = []
        self.weights_history = []
        self.selected_features_history = []

        # This indicator will be set to True, if the feature weights returned by the feature selector contain negative
        # values. In that case, we raise a warning once, since we use the absolute weights for feature selection.
        self._scale_warning_issued = False

    @abstractmethod
    def weight_features(self, X: ArrayLike, y: ArrayLike):
        """Updates feature weights.

        Args:
            X: Array/matrix of observations.
            y: Array of corresponding labels.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """Resets the feature selector."""
        raise NotImplementedError

    def select_features(self, X: ArrayLike, rng: Generator) -> ArrayLike:
        """Selects features with highest absolute weights.

        Args:
            X: Array/matrix of observations.
            rng: A numpy random number generator object.

        Returns:
            ArrayLike: The observations with all non-selected features replaced by the baseline value.
        """
        if np.any(self.weights < 0):
            abs_weights = abs(self.weights)
            if not self._scale_warning_issued:
                warnings.warn("The weight vector contains negative values. The absolute weights will be used for "
                              "feature selection.")
                self._scale_warning_issued = True
        else:
            abs_weights = self.weights

        sorted_indices = np.argsort(abs_weights)[::-1]
        self.selected_features = sorted_indices[:self.n_selected_features]
        self.weights_history.append(abs_weights.tolist())
        self.selected_features_history.append(self.selected_features.tolist())

        X_new = self._get_baseline(X=X, rng=rng)
        X_new[:, self.selected_features] = X[:, self.selected_features]

        return X_new

    def _get_baseline(self, X: ArrayLike, rng: Generator) -> ArrayLike:
        """Returns a matrix/vector filled with the baseline.

        Please cite:
        Haug, Johannes, et al. "On Baselines for Local Feature Attributions." arXiv preprint arXiv:2101.00905 (2021).

        Args:
            X: Array/matrix of observations.
            rng: A numpy random number generator object.

        Returns:
            ArrayLike: A matrix in the shape of x, pre-filled with the baseline.
        """
        if self.baseline == 'constant':
            # Constant baseline value
            if not isinstance(self.ref_sample, (int, float)):
                warnings.warn("No integer value provided via ref_sample. Baseline 'constant' will return zero.")
                return np.zeros_like(X)
            return np.ones_like(X) * self.ref_sample

        elif self.baseline == 'max_dist':
            # Baseline equals reference observation with max. euclidean distance regarding a given instance
            X_new = np.zeros_like(X)
            for i, x in enumerate(X):
                dist = [np.linalg.norm(x - x_ref) for x_ref in self.ref_sample]
                X_new[i, :] = self.ref_sample[np.argmax(dist), :]
            return X_new

        elif self.baseline == 'gaussian':
            # Baseline is sampled from feature-wise Gaussian distributions (loc and scale acc. to ref sample)
            X_new = np.zeros_like(X)
            for ftr in range(X.shape[1]):
                loc = np.mean(self.ref_sample[:, ftr], axis=0)
                scale = np.std(self.ref_sample[:, ftr], axis=0)
                X_new[:, ftr] = rng.normal(loc=loc, scale=scale, size=X_new.shape[0])
            return X_new

        elif self.baseline == 'uniform':
            # Baseline is sampled from feature-wise Uniform distributions (low and high acc. to ref sample)
            X_new = np.zeros_like(X)
            for ftr in range(X.shape[1]):
                low = np.min(self.ref_sample[:, ftr], axis=0)
                high = np.max(self.ref_sample[:, ftr], axis=0)
                X_new[:, ftr] = rng.uniform(low=low, high=high, size=X_new.shape[0])
            return X_new

        elif self.baseline == 'expectation':
            # Baseline equals the sample expectation
            return np.tile(np.mean(self.ref_sample, axis=0), (X.shape[0], 1))

        else:
            warnings.warn(
                "Baseline method {} is not implemented. We use the 'zero' baseline instead.".format(self.baseline))
            return np.zeros_like(X)
