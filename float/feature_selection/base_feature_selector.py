from abc import ABCMeta, abstractmethod
import numpy as np
import warnings


class BaseFeatureSelector(metaclass=ABCMeta):
    """
    Abstract base class for online feature selection methods.

    Attributes:
        n_total_features (int): total number of features
        n_selected_features (int): number of selected features
        supports_multi_class (bool): True if model support multi-class classification, False otherwise
        raw_weight_vector (np.ndarray): current weights (as produced by feature selection model)
        weights (list): absolute weights in all time steps
        selection (list): indices of selected features in all time steps
        baseline (str): identifier of baseline method (value to replace non-selected features with)
        ref_sample (float | np.ndarray): sample used to obtain the baseline (not required for 'zero' baseline)
    """

    def __init__(self, n_total_features, n_selected_features, supports_multi_class, reset_after_drift, baseline, ref_sample):
        """
        Receives parameters of feature selection model.

        Args:
            n_total_features (int): total number of features
            n_selected_features (int): number of selected features
            supports_multi_class (bool): True if model support multi-class classification, False otherwise
            reset_after_drift (bool): indicates whether to reset the predictor after a drift was detected
            baseline (str): identifier of baseline method (value to replace non-selected features with)
            ref_sample (float | np.ndarray): integer (in case of 'constant' baseline) or sample used to obtain the baseline
        """
        self.reset_after_drift = reset_after_drift
        self.n_total_features = n_total_features
        self.n_selected_features = n_selected_features
        self.baseline = baseline
        self.ref_sample = ref_sample

        self.supports_multi_class = supports_multi_class

        self.raw_weight_vector = np.zeros(self.n_total_features)
        self.weights = []
        self.selection = []
        self.selected_features = []
        self._auto_scale = False

    @abstractmethod
    def weight_features(self, X, y):
        """
        Given a batch of observations and corresponding labels, computes feature weights.

        Args:
            X (np.ndarray): samples of current batch
            y (np.ndarray): labels of current batch
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """
        Reset the feature selector.
        """
        raise NotImplementedError

    def select_features(self, X):
        """
        Selects features with highest absolute weights.

        Args:
            X (np.ndarray): the data samples

        Returns:
            np.ndarray: the data samples with the non-selected features set to a reference value
        """
        if np.any(self.raw_weight_vector < 0):
            abs_weights = abs(self.raw_weight_vector)
            if not self._auto_scale:
                warnings.warn('Weight vector contains negative weights. Absolute weights will be used for feature'
                              ' selection.')
                self._auto_scale = True
        else:
            abs_weights = self.raw_weight_vector

        sorted_indices = np.argsort(abs_weights)[::-1]
        self.selected_features = sorted_indices[:self.n_selected_features]
        self.weights.append(abs_weights.tolist())
        self.selection.append(self.selected_features.tolist())

        X_new = self._get_baseline(X)
        X_new[:, self.selected_features] = X[:, self.selected_features]
        return X_new

    def _get_baseline(self, X):
        """
        Returns the baseline value to be used for the non-selected features.

        Please cite:
        Haug, Johannes, et al. "On Baselines for Local Feature Attributions." arXiv preprint arXiv:2101.00905 (2021).

        Args:
            X (np.ndarray): the data samples

        Returns:
            np.array: a matrix in the shape of X, which is pre-filled with the baselines
        """
        rng = np.random.default_rng(0)  # Todo: use global rng

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
            warnings.warn("Baseline method {} is not implemented. We use 'zero' baseline instead.".format(self.baseline))
