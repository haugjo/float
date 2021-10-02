from float.feature_selection.base_feature_selector import BaseFeatureSelector
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class EFS(BaseFeatureSelector):
    """
    Extremal Feature Selection.

    Based on a paper by Carvalho et al. 2005. This Feature Selection algorithm is based on the weights of a
    Modified Balanced Winnow classifier (as introduced in the paper).
    """
    def __init__(self, n_total_features, n_selected_features, u=None, v=None, theta=1, M=1,
                 alpha=1.5, beta=0.5, reset_after_drift=False, baseline='constant', ref_sample=0):
        """
        Initializes the EFS feature selector.

        Args:
            n_total_features (int): total number of features
            n_selected_features (int): number of selected features
            u (np.ndarray): initial positive model with weights set to 2
            v (np.ndarray): initial negative model with weights
            theta (float): threshold parameter
            M (float): margin parameter
            alpha (float): promotion parameter
            beta (float): demotion parameter
            reset_after_drift (bool): indicates whether to reset the predictor after a drift was detected
            baseline (str): identifier of baseline method (value to replace non-selected features with)
            ref_sample (float | np.array): sample used to obtain the baseline (not required for 'zero' baseline)
        """
        super().__init__(n_total_features, n_selected_features, supports_multi_class=False,
                         reset_after_drift=reset_after_drift, baseline=baseline, ref_sample=ref_sample)

        self.u_init = u
        self.v_init = v
        self.u = np.ones(n_total_features) * 2 if u is None else u
        self.v = np.ones(n_total_features) if v is None else v

        self.theta = theta
        self.M = M
        self.alpha = alpha
        self.beta = beta

    def weight_features(self, X, y):
        """
        Given a batch of observations and corresponding labels, computes feature weights.

        Args:
            X (np.ndarray): samples of current batch
            y (np.ndarray): labels of current batch
        """
        # iterate over all elements in batch
        for x_b, y_b in zip(X, y):

            # Convert label to -1 and 1
            y_b = -1 if y_b == 0 else 1

            # Note, the original algorithm here adds a "bias" feature that is always 1

            # Normalize x_b
            x_b = MinMaxScaler().fit_transform(x_b.reshape(-1, 1)).flatten()

            # Calculate score
            score = np.dot(x_b, self.u) - np.dot(x_b, self.v) - self.theta

            # If prediction was mistaken
            if score * y_b <= self.M:
                # Update models for all features j
                for j, _ in enumerate(self.u):
                    if y_b > 0:
                        self.u[j] = self.u[j] * self.alpha * (1 + x_b[j])
                        self.v[j] = self.v[j] * self.beta * (1 - x_b[j])
                    else:
                        self.u[j] = self.u[j] * self.beta * (1 - x_b[j])
                        self.v[j] = self.v[j] * self.alpha * (1 + x_b[j])

        # Compute importance score of features
        self.raw_weight_vector = abs(self.u - self.v)

    def reset(self):
        """
        Reset Winnow weights
        """
        self.u = np.ones(self.n_total_features) * 2 if self.u_init is None else self.u_init
        self.v = np.ones(self.n_total_features) if self.v_init is None else self.v_init
