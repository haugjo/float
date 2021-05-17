from abc import ABCMeta, abstractmethod
import numpy as np
import warnings


class FeatureSelector(metaclass=ABCMeta):
    """
    Abstract base class for online feature selection methods.

    Attributes:
        name (str): name of feature selection model
        n_total_ftr (int): total number of features
        n_selected_ftr (int): number of selected features
        supports_multi_class (bool): True if model support multi-class classification, False otherwise
        supports_streaming_features (bool): True if model supports streaming features, False otherwise
        raw_weight_vector (np.ndarray): current weights (as produced by feature selection model)
        weights (list): absolute weights in all time steps
        selection (list): indices of selected features in all time steps
        comp_time (list): computation time in all time steps
        _auto_scale (bool): indicator for scaling of weights
    """

    def __init__(self, name, n_total_ftr, n_selected_ftr, supports_multi_class=False, supports_streaming_features=False):
        """
        Receives parameters of feature selection model.

        Args:
            name (str): name of feature selection model
            n_total_ftr (int): total number of features
            n_selected_ftr (int): number of selected features
            supports_multi_class (bool): True if model support multi-class classification, False otherwise
            supports_streaming_features (bool): True if model supports streaming features, False otherwise
        """
        self.name = name
        self.n_total_ftr = n_total_ftr
        self.n_selected_ftr = n_selected_ftr
        self.supports_multi_class = supports_multi_class
        self.supports_streaming_features = supports_streaming_features

        self.raw_weight_vector = np.zeros(self.n_total_ftr)
        self.weights = []
        self.selection = []
        self.comp_time = []
        self._auto_scale = False

    @abstractmethod
    def weight_features(self, X, y):
        """
        Weights features.

        Args:
            X (np.ndarray): samples of current batch
            y (np.ndarray): labels of current batch
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
        # if vector contains negative weights, issue warning
        if np.any(self.raw_weight_vector < 0):
            abs_weights = abs(self.raw_weight_vector)
            if not self._auto_scale:
                warnings.warn('Weight vector contains negative weights. Absolute weights will be used for feature'
                              ' selection.')
                self._auto_scale = True
        else:
            abs_weights = self.raw_weight_vector

        sorted_indices = np.argsort(abs_weights)[::-1]
        selected_indices = sorted_indices[:self.n_selected_ftr]
        non_selected_indices = sorted_indices[self.n_selected_ftr:]
        X[:, non_selected_indices] = np.full(shape=X[:, non_selected_indices].shape, fill_value=self._get_reference_value())

        self.weights.append(abs_weights.tolist())
        self.selection.append(selected_indices.tolist())

        return X

    def _get_reference_value(self):
        """
        Returns the reference value to be used for the non-selected features.

        Returns:
            float: the reference value
        """
        return 0
