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
        comp_times (list): computation time in all time steps
    """

    def __init__(self, n_total_features, n_selected_features, supports_multi_class, reset_after_drift):
        """
        Receives parameters of feature selection model.

        Args:
            n_total_features (int): total number of features
            n_selected_features (int): number of selected features
            supports_multi_class (bool): True if model support multi-class classification, False otherwise
            reset_after_drift (bool): indicates whether to reset the predictor after a drift was detected
        """
        self.reset_after_drift = reset_after_drift
        self.n_total_features = n_total_features
        self.n_selected_features = n_selected_features

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

        X_new = np.full(X.shape, self._get_reference_value())
        X_new[:, self.selected_features] = X[:, self.selected_features]
        return X_new

    def _get_reference_value(self):
        """
        Returns the reference value to be used for the non-selected features.

        Returns:
            float: the reference value
        """
        return 0
