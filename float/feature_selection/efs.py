from float.feature_selection.feature_selector import FeatureSelector
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class EFS(FeatureSelector):
    """
    Extremal Feature Selection.

    Based on a paper by Carvalho et al. 2005. This Feature Selection algorithm is based on the weights of a
    Modified Balanced Winnow classifier (as introduced in the paper).
    """
    def __init__(self, n_total_features, n_selected_features, u=None, v=None, threshold=1, M=1, alpha=1.5, beta=0.5):
        super().__init__(n_total_features, n_selected_features, supports_multi_class=False, supports_streaming_features=False)

        if u is None:
            self.u = np.ones(n_total_features) * 2  # initial positive model with weights 2
        else:
            self.u = u
        if v is None:
            self.v = np.ones(n_total_features)  # initial negative model with weights
        else:
            self.v = v

        self.threshold = threshold  # threshold parameter
        self.M = M  # margin
        self.alpha = alpha  # promotion parameter
        self.beta = beta  # demotion parameter

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
            score = np.dot(x_b, self.u) - np.dot(x_b, self.v) - self.threshold

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
