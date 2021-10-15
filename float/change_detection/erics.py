"""ERICS Change Detection Method.

The ERICS (Effective and Robust Identification of Concept Shift) change detector was proposed by:
[1] HAUG, Johannes; KASNECI, Gjergji. Learning Parameter Distributions to Detect Concept Drift in Data Streams.
    In: 2020 25th International Conference on Pattern Recognition (ICPR). IEEE, 2021. S. 9452-9459.

The original source code can be obtained here: https://github.com/haugjo/erics

This module provides the ERICS implementation with a Probit base model for binary classification scenarios.
The update rules for the Probit model are adopted from:
[2] HAUG, Johannes, et al. Leveraging model inherent variable importance for stable online feature selection.
    In: Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2020.
    S. 1478-1502.

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
import copy
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm
from typing import Tuple
from warnings import warn

from float.change_detection.base_change_detector import BaseChangeDetector


class ERICS(BaseChangeDetector):
    """ERICS Change Detector."""
    def __init__(self, n_param: int, window_mvg_average: int = 50, window_drift_detect: int = 50, beta: float = 0.0001,
                 init_mu: int = 0, init_sigma: int = 1, epochs: int = 10, lr_mu: float = 0.01, lr_sigma: float = 0.01,
                 reset_after_drift: bool = False):
        """Inits the ERICS change detector.

        Args:
            n_param:
                The total number of parameters in the Probit model. This corresponds to the number of input features.
            window_mvg_average:
                he window size for the computation of the moving average of the KL divergence.
            window_drift_detect: The window size that is used to compute the pairwise differences between
                subsequent measures of the moving average. This is information is used for the actual change detection.
            beta: The Update rate of the alpha-threshold.
            init_mu: The initial mean of the parameter distributions.
            init_sigma: The initial variance of the parameter distributions.
            epochs: The number of epochs per the optimization iteration of the parameter distributions.
            lr_mu: The learning rate for the gradient updates of the mean.
            lr_sigma: The learning rate for the gradient updates of the variance.
            reset_after_drift: See description of base class.
        """
        if reset_after_drift:
            warn("The ERICS change detector need not be reset after detecting a concept drift. "
                 "We set reset_after_drift to False.")

        super().__init__(reset_after_drift=False, error_based=False)

        self._n_param = n_param
        self._M = window_mvg_average
        self._W = window_drift_detect
        self._beta = beta

        self._time_step = 0                                             # Current Time Step
        self._time_since_last_drift = 0                                 # Time steps since last global drift detection
        self._time_since_last_partial_drift = np.zeros(n_param)         # Time steps since last partial drift detection
        self._alpha = None                                              # Threshold for global concept drift detection
        self._partial_alpha = np.asarray([None] * self._n_param)        # Threshold for partial concept drift detection
        self._mu_w = np.ones((self._M, self._n_param)) * init_mu        # Parameter Mean in window
        self._sigma_w = np.ones((self._M, self._n_param)) * init_sigma  # Parameter Variance in window
        self._param_sum = np.zeros((self._M - 1, self._n_param))        # Sum-expression for moving average
        self._info_ma = []                                              # Global moving average
        self._partial_info_ma = []                                      # Partial moving average

        # Probit model (acc. to the FIRES online feature selection framework [2])
        self._fires_mu = np.ones(self._n_param) * init_mu
        self._fires_sigma = np.ones(self._n_param) * init_sigma
        self._fires_epochs = epochs
        self._fires_lr_mu = lr_mu
        self._fires_lr_sigma = lr_sigma
        self._fires_labels = []  # Unique labels (Note: fires requires binary labels)

        self._drift_detected = False
        self._partial_drift_detected = False
        self._partial_drift_features = None

    def reset(self):
        """Resets the change detector.

        Notes:
            ERICS need not be reset after a drift was detected.
        """
        pass

    def partial_fit(self, X: ArrayLike, y: ArrayLike):
        """Updates the change detector.

        Args:
            X: Batch of observations.
            y: Batch of labels.
        """
        self._drift_detected = False

        # Update alpha (Eq. 7 in [1])
        if self._alpha is not None:
            self._alpha -= (self._alpha * self._beta * self._time_since_last_drift)
        for k in range(self._n_param):
            if self._partial_alpha[k] is not None:
                self._partial_alpha[k] -= (self._partial_alpha[k] * self._beta * self._time_since_last_partial_drift[k])

        # Update time since drift
        self._time_since_last_drift += 1
        self._time_since_last_partial_drift += 1

        self._update_probit(X=X, y=y)          # Update Parameter distribution
        self._update_param_sum()           # Update the sum expression for observations in a shifting window
        self._compute_moving_average()     # Compute moving average in specified window

        self._drift_detected, self._partial_drift_detected, self._partial_drift_features = self._detect_drift()

        # Update time step
        self._time_step += 1

    def detect_change(self) -> bool:
        """Detects global concept drift."""
        return self._drift_detected

    def detect_partial_change(self) -> Tuple[bool, list]:
        """Detects partial concept drift (see base class)."""
        return self._partial_drift_detected, self._partial_drift_features

    def detect_warning_zone(self) -> bool:
        """Detects a warning zone.

        Notes:
            ERICS does not raise warnings.
        """
        return False

    # ----------------------------------------
    # ERICS Functionality (left unchanged)
    # ----------------------------------------
    def _update_param_sum(self):
        """Retrieve current parameter distribution and compute sum expression according to Eq. (8) [1]."""
        # Retrieve current distribution parameters
        new_mu = copy.copy(self._fires_mu).reshape(1, -1)
        new_sigma = copy.copy(self._fires_sigma).reshape(1, -1)

        # Drop oldest entry from window
        self._mu_w = self._mu_w[1:, :]
        self._sigma_w = self._sigma_w[1:, :]

        # Add new entry to window
        self._mu_w = np.concatenate((self._mu_w, new_mu))
        self._sigma_w = np.concatenate((self._sigma_w, new_sigma))

        # Compute parameter sum expression
        for t in range(self._M - 1):
            self._param_sum[t, :] = (self._sigma_w[t + 1, :] ** 2 + (self._mu_w[t, :] - self._mu_w[t + 1, :]) ** 2) / \
                                    self._sigma_w[t, :] ** 2

    def _compute_moving_average(self):
        """Compute the moving average (according to Eq. (8) [1])."""
        partial_ma = np.zeros(self._n_param)
        score = np.zeros(self._M - 1)

        for k in range(self._n_param):
            partial_score = self._param_sum[:, k] - 1
            score += partial_score
            partial_ma[k] = np.sum(np.abs(partial_score)) / (2 * self._M)  # Add partial mov. avg. for parameter k

        ma = np.sum(np.abs(score)) / (2 * self._M)

        self._info_ma.append(ma)
        self._partial_info_ma.append(partial_ma)

    def _detect_drift(self) -> Tuple[bool, bool, list]:
        """Detect global and partial concept drift using the adaptive alpha-threshold

        Returns:
            bool: True, if a concept drift was detected, False otherwise.
            bool: True, if a partial concept drift was detected, False otherwise.
            list: Indices of input features with detected partial drift.
        """
        window_delta = None
        partial_window_delta = None

        # Compute delta in sliding window W (according to Eq. (5) in the ERICS paper [1])
        if self._W < 2:
            self._W = 2
            warn('Sliding window for concept drift detection was automatically set to 2 observations.')

        if len(self._info_ma) < self._W:
            oldest_entry = len(self._info_ma)
        else:
            oldest_entry = self._W

        if oldest_entry == 1:  # In case of only one observation
            window_delta = copy.copy(self._info_ma[-1])
            partial_window_delta = copy.copy(self._partial_info_ma[-1])
        else:
            for t in range(oldest_entry, 1, -1):
                if t == oldest_entry:
                    window_delta = self._info_ma[-t + 1] - self._info_ma[-t]  # newer - older
                    partial_window_delta = self._partial_info_ma[-t+1] - self._partial_info_ma[-t]
                else:
                    window_delta += (self._info_ma[-t + 1] - self._info_ma[-t])
                    partial_window_delta += (self._partial_info_ma[-t+1] - self._partial_info_ma[-t])

        # (Re-) Initialize alpha if it is None (at time step 0 or if a drift was detected)
        if self._alpha is None:
            # According to Eq. (6) in [1] -> abs() is only required at t=0, to make sure that alpha > 0
            self._alpha = np.abs(window_delta)
        if None in self._partial_alpha:
            unspecified = np.isnan(self._partial_alpha.astype(float)).flatten()
            self._partial_alpha[unspecified] = np.abs(partial_window_delta[unspecified])

        # Drift Detection
        drift = False
        if window_delta > self._alpha:
            drift = True
            self._time_since_last_drift = 0
            self._alpha = None

        # Partial Drift Detection
        p_drift = False
        partial_drift_bool = partial_window_delta > self._partial_alpha
        partial_drift_features = np.argwhere(partial_drift_bool).tolist()
        for k in partial_drift_features:
            p_drift = True
            self._time_since_last_partial_drift[k] = 0
            self._partial_alpha[k] = None

        return drift, p_drift, partial_drift_features

    def _update_probit(self, X: np.ndarray, y: np.ndarray):
        """Update parameters of the Probit model.

        According to [2], as implemented here https://github.com/haugjo/fires
        We have slightly adjusted the original code to fit our use case.

        Args:
            X: Batch of observations.
            y: Batch of labels. The labels must be binary and will be automatically encoded as {-1,1}

        Raises:
            ValueError: If the target is not a binary variable.
            TypeError: If input features are not numeric.
        """
        # Encode labels
        for y_val in np.unique(y):  # Add newly observed unique labels
            if y_val not in set(self._fires_labels):
                self._fires_labels.append(y_val)

        if tuple(self._fires_labels) != (-1, 1):  # Check if labels are encoded correctly
            if len(self._fires_labels) < 2:
                y[y == self._fires_labels[0]] = -1
            elif len(self._fires_labels) == 2:
                y[y == self._fires_labels[0]] = -1
                y[y == self._fires_labels[1]] = 1
            else:
                raise ValueError("The target variable y must be binary.")

        for epoch in range(self._fires_epochs):
            # Shuffle the observations
            random_idx = np.random.permutation(len(y))
            X = X[random_idx]
            y = y[random_idx]

            # Iterative update of mu and sigma
            try:
                # Helper functions
                dot_mu_x = np.dot(X, self._fires_mu)
                rho = np.sqrt(1 + np.dot(X ** 2, self._fires_sigma ** 2))

                # Gradients
                nabla_mu = norm.pdf(y / rho * dot_mu_x) * (y / rho * X.T)
                nabla_sigma = norm.pdf(y / rho * dot_mu_x) * (
                            - y / (2 * rho ** 3) * 2 * (X ** 2 * self._fires_sigma).T * dot_mu_x)

                # Marginal Likelihood
                marginal = norm.cdf(y / rho * dot_mu_x)

                # Update parameters
                self._fires_mu += self._fires_lr_mu * np.mean(nabla_mu / marginal, axis=1)
                self._fires_sigma += self._fires_lr_sigma * np.mean(nabla_sigma / marginal, axis=1)
            except TypeError as e:
                raise TypeError("All features must be a numeric data type.") from e
