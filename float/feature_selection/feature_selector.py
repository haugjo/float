from abc import ABCMeta, abstractmethod
import numpy as np
import warnings
import traceback


class FeatureSelector(metaclass=ABCMeta):
    """
    Abstract base class for online feature selection methods.

    Attributes:
        n_total_features (int): total number of features
        n_selected_features (int): number of selected features
        evaluation (dict of str: list[float]): a dictionary of metric names and their corresponding metric values as lists
        supports_multi_class (bool): True if model support multi-class classification, False otherwise
        supports_streaming_features (bool): True if model supports streaming features, False otherwise
        raw_weight_vector (np.ndarray): current weights (as produced by feature selection model)
        weights (list): absolute weights in all time steps
        selection (list): indices of selected features in all time steps
        comp_times (list): computation time in all time steps
    """

    def __init__(self, n_total_features, n_selected_features, evaluation_metrics, supports_multi_class,
                 supports_streaming_features, streaming_features=None):
        """
        Receives parameters of feature selection model.

        Args:
            n_total_features (int): total number of features
            n_selected_features (int): number of selected features
            evaluation_metrics (dict of str: function | dict of str: (function, dict)): {metric_name: metric_function} OR {metric_name: (metric_function, {param_name1: param_val1, ...})} a dictionary of metrics to be used
            supports_multi_class (bool): True if model support multi-class classification, False otherwise
            supports_streaming_features (bool): True if model supports streaming features, False otherwise
            streaming_features (dict): (time, feature index) tuples to simulate streaming features
        """
        self.n_total_features = n_total_features
        self.n_selected_features = n_selected_features
        self.evaluation_metrics = evaluation_metrics if evaluation_metrics else {'Nogueira Stability Measure': (FeatureSelector.get_nogueira_stability, {'n_total_features': self.n_total_features, 'nogueira_window_size': 10})}
        self.evaluation = {key: [] for key in self.evaluation_metrics.keys()}

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

    def evaluate(self):
        """
        Evaluates the feature selector at one time step.
        """
        for metric_name in self.evaluation:
            if isinstance(self.evaluation_metrics[metric_name], tuple):
                metric_func = self.evaluation_metrics[metric_name][0]
                metric_params = self.evaluation_metrics[metric_name][1]
            else:
                metric_func = self.evaluation_metrics[metric_name]
                metric_params = {}
            try:
                metric_val = metric_func(self.selection, **metric_params)
            except TypeError:
                # TODO include names of missing parameters in warning message
                warnings.warn(f'{metric_name} will not be evaluated because of one or more missing function parameters.')
                continue

            self.evaluation[metric_name].append(metric_val)

    @staticmethod
    def get_nogueira_stability(selection, n_total_features, nogueira_window_size=10):
        """
        Returns the Nogueira measure for feature selection stability.

        Returns:
            float: the stability measure
        """
        Z = np.zeros([min(len(selection), nogueira_window_size), n_total_features])
        for row, col in enumerate(selection[-nogueira_window_size:]):
            Z[row, col] = 1

        try:
            M, d = Z.shape
            hatPF = np.mean(Z, axis=0)
            kbar = np.sum(hatPF)
            denom = (kbar / d) * (1 - kbar / d)
            stability_measure = 1 - (M / (M - 1)) * np.mean(np.multiply(hatPF, 1 - hatPF)) / denom
        except ZeroDivisionError:
            stability_measure = 0  # metric requires at least 2 measurements and thus runs an error at t=1

        return stability_measure

    def _get_reference_value(self):
        """
        Returns the reference value to be used for the non-selected features.

        Returns:
            float: the reference value
        """
        return 0
