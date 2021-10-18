"""FIRES Feature Selection Method.

This module contains the Fast, Interpretable and Robust Evaluation and Selection of features (FIRES) with a Probit base
model and normally distributed parameters as introduced by:
HAUG, Johannes, et al. Leveraging model inherent variable importance for stable online feature selection.
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
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
from typing import Union
from warnings import warn

from float.feature_selection.base_feature_selector import BaseFeatureSelector


class FIRES(BaseFeatureSelector):
    """FIRES feature selector."""
    def __init__(self,
                 n_total_features: int,
                 n_selected_features: int,
                 classes: list,
                 mu_init: Union[int, ArrayLike] = 0,
                 sigma_init: Union[int, ArrayLike] = 1,
                 penalty_s: float = 0.01,
                 penalty_r: float = 0.01,
                 epochs: int = 1,
                 lr_mu: float = 0.01,
                 lr_sigma: float = 0.01,
                 scale_weights: bool = True,
                 reset_after_drift: bool = False,
                 baseline: str = 'constant',
                 ref_sample: Union[float, ArrayLike] = 0):
        """Inits the feature selector.

        Args:
            n_total_features: See description of base class.
            n_selected_features: See description of base class.
            classes: A list of unique target values (class labels).
            mu_init:
                Initial importance, i.e. mean of the parameter distribution. One may either set the initial values
                separately per feature (by providing a vector), or use the same initial value for all features
                (by providing a scalar).
            sigma_init:
                Initial uncertainty, i.e. standard deviation of the parameter distribution. One may either set the
                initial values separately per feature (by providing a vector), or use the same initial value for all
                features (by providing a scalar).
            penalty_s: Penalty factor in the optimization of weights w.r.t the uncertainty (corresponds to gamma_s in
                the paper).
            penalty_r : Penalty factor in the optimization of weights for the regularization (corresponds to gamma_r
                in the paper).
            epochs: Number of epochs in each update iteration.
            lr_mu: Learning rate for the gradient update of the mean.
            lr_sigma: Learning rate for the gradient update of the standard deviation.
            scale_weights: If True, scale feature weights into the range [0,1]. If False, do not scale weights.
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

        self._n_total_ftr = n_total_features
        self._classes = classes
        self._mu_init = mu_init
        self._sigma_init = sigma_init
        self._mu = np.ones(n_total_features) * mu_init
        self._sigma = np.ones(n_total_features) * sigma_init
        self._penalty_s = penalty_s
        self._penalty_r = penalty_r
        self._epochs = epochs
        self._lr_mu = lr_mu
        self._lr_sigma = lr_sigma
        self._scale_weights = scale_weights

    def weight_features(self, X: ArrayLike, y: ArrayLike):
        """Updates feature weights."""
        # Update estimates of mu and sigma given the predictive model
        self._probit(X=X, y=y)

        # Limit sigma to range [0, inf]
        if sum(n < 0 for n in self._sigma) > 0:
            self._sigma[self._sigma < 0] = 0
            warn('Sigma has automatically been rescaled to [0, inf], because it contained negative values.')

        # Compute feature weights
        self.weights = self._compute_weights()

    def reset(self):
        """Resets the feature selector."""
        self._mu = np.ones(self.n_total_features) * self._mu_init
        self._sigma = np.ones(self.n_total_features) * self._sigma_init

    # ----------------------------------------
    # FIRES Functionality
    # ----------------------------------------
    def _probit(self, X: ArrayLike, y: ArrayLike):
        """Updates the Probit model parameters.

        This function updates the distribution parameters mu and sigma by optimizing them in terms of the (log)
        likelihood. Here we assume a Bernoulli distributed target variable. We use a Probit model as our base model.
        This corresponds to the FIRES-GLM model in the paper.

        Args:
            X: Array/matrix of observations.
            y: Array of corresponding labels.

        Raises:
            TypeError: If features are not in numeric type.
        """
        # Encode labels
        for y_val in np.unique(y):  # Add newly observed unique labels
            if y_val not in set(self._classes):
                self._classes.append(y_val)

        if tuple(self._classes) != (-1, 1):  # Check if labels are encoded correctly
            if len(self._classes) < 2:
                y[y == self._classes[0]] = -1
            elif len(self._classes) == 2:
                y[y == self._classes[0]] = -1
                y[y == self._classes[1]] = 1
            else:
                raise ValueError('The target variable y must be binary.')

        for epoch in range(self._epochs):
            # Shuffle the observations
            random_idx = np.random.permutation(len(y))
            X = X[random_idx]
            y = y[random_idx]

            # Iterative update of mu and sigma
            try:
                # Helper functions
                dot_mu_x = np.dot(X, self._mu)
                rho = np.sqrt(1 + np.dot(X ** 2, self._sigma ** 2))

                # Gradients
                nabla_mu = norm.pdf(y / rho * dot_mu_x) * (y / rho * X.T)
                nabla_sigma = norm.pdf(y / rho * dot_mu_x) * (
                        - y / (2 * rho ** 3) * 2 * (X ** 2 * self._sigma).T * dot_mu_x)

                # Marginal Likelihood
                marginal = norm.cdf(y / rho * dot_mu_x)

                # Update parameters
                self._mu += self._lr_mu * np.mean(nabla_mu / marginal, axis=1)
                self._sigma += self._lr_sigma * np.mean(nabla_sigma / marginal, axis=1)
            except TypeError as e:
                raise TypeError("All features must be a numeric data type.") from e

    def _compute_weights(self) -> ArrayLike:
        """Computes feature weights from the Probit model parameters.

        Computes optimal weights according to the objective function proposed in the paper.
        We compute feature weights in a trade-off between feature importance and uncertainty.
        Thereby, we aim to maximize both the discriminative power and the stability/robustness of feature weights.

        Returns:
            ArrayLike: FIRES feature weights.
        """
        # Compute optimal weights
        weights = (self._mu ** 2 - self._penalty_s * self._sigma ** 2) / (2 * self._penalty_r)

        if self._scale_weights:  # Scale weights to [0,1]
            weights = MinMaxScaler().fit_transform(weights.reshape(-1, 1)).flatten()

        return weights
