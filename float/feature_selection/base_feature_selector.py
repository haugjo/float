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
        supports_streaming_features (bool): True if model supports streaming features, False otherwise
        raw_weight_vector (np.ndarray): current weights (as produced by feature selection model)
        weights (list): absolute weights in all time steps
        selection (list): indices of selected features in all time steps
        comp_times (list): computation time in all time steps
    """

    def __init__(self, n_total_features, n_selected_features, supports_multi_class, supports_streaming_features,
                 streaming_features, reset_after_drift):
        """
        Receives parameters of feature selection model.

        Args:
            n_total_features (int): total number of features
            n_selected_features (int): number of selected features
            supports_multi_class (bool): True if model support multi-class classification, False otherwise
            supports_streaming_features (bool): True if model supports streaming features, False otherwise
            streaming_features (dict | None): (time, feature index) tuples to simulate streaming features
            reset_after_drift (bool): indicates whether to reset the predictor after a drift was detected
        """
        self.reset_after_drift = reset_after_drift
        self.n_total_features = n_total_features
        self.n_selected_features = n_selected_features

        self.supports_multi_class = supports_multi_class
        self.supports_streaming_features = supports_streaming_features
        self.streaming_features = streaming_features if streaming_features else dict()

        self.raw_weight_vector = np.zeros(self.n_total_features)
        self.weights = []
        self.selection = []
        self.comp_times = []
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

    def select_features(self, X, time_step):
        """
        Selects features with highest absolute weights.

        Args:
            X (np.ndarray): the data samples
            time_step (int): the current time step

        Returns:
            np.ndarray: the data samples with the non-selected features set to a reference value
        """
        if self.supports_streaming_features:
            if time_step == 0 and time_step not in self.streaming_features:
                self.selected_features = np.arange(self.n_total_features)
                warnings.warn(
                    'Simulate streaming features: No active features provided at t=0. All features are used instead.')
            elif time_step in self.streaming_features:
                self.selected_features = self.streaming_features[time_step]
        else:
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
