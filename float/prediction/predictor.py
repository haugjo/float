from abc import ABCMeta, abstractmethod
import numpy as np


class Predictor(metaclass=ABCMeta):
    """
    Abstract base class which serves as both a wrapper for skmultiflow predictive models and a constructor for
    user-defined predictive models.

    Attributes:
        evaluation (dict of str: list[float]): a dictionary of metric names and their corresponding metric values as lists
        testing_times (list): testing times per time step
        training_times (list): training times per time step
    """

    @abstractmethod
    def __init__(self, evaluation_metrics, decay_rate, window_size):
        """
        Initializes the predictor.

        Args:
            evaluation_metrics (dict of str: function | dict of str: (function, dict)): {metric_name: metric_function} OR {metric_name: (metric_function, {param_name1: param_val1, ...})} a dictionary of metrics to be used
            decay_rate (float): when this parameter is set, the metric values are additionally stored in a decayed version
            window_size (int): when this parameter is set, the metric values are additionally stored in a sliding window version
        """
        self.decay_rate = decay_rate
        self.window_size = window_size

        self.evaluation_metrics = evaluation_metrics
        self.evaluation = {key: [] for key in self.evaluation_metrics.keys()} if evaluation_metrics else {}
        if self.decay_rate:
            self.evaluation_decay = {key: [] for key in self.evaluation_metrics.keys()} if evaluation_metrics else {}
        if self.window_size:
            self.evaluation_window = {key: [] for key in self.evaluation_metrics.keys()} if evaluation_metrics else {}

        self.testing_times = []
        self.training_times = []

    def fit(self, X, y, sample_weight=None):
        """
        Fits the model.

        Args:
            X (np.ndarray): data samples to train the model with
            y (np.ndarray): target values for all samples in X
            sample_weight (np.ndarray): sample weights

        Returns:
            self
        """
        self.partial_fit(X, y, sample_weight=sample_weight)

        return self

    @abstractmethod
    def partial_fit(self, X, y, sample_weight=None):
        """
        Partially (incrementally) fit the model.

        Args:
            X (np.ndarray): data samples to train the model with
            y (np.ndarray): target values for all samples in X
            sample_weight (np.ndarray): sample weights

        Returns:
            self
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        """
        Predicts target values for the passed data.

        Args:
            X (np.ndarray): data samples to predict the values for

        Returns:
            np.ndarray: predictions for all samples in X
        """
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X):
        """
        Estimates the probability for probabilistic/bayesian regressors.

        Args:
            X (np.ndarray): data samples to predict the probabilities for

        Returns:
            np.ndarray: prediction probability for all samples in X
        """
        raise NotImplementedError

    def evaluate(self, y_pred, y):
        """
        Evaluates the predictor at one time step.

        Args:
            y_pred (np.ndarray): predicted labels for the data samples
            y (np.ndarray): true values for the data samples
        """
        for metric_name in self.evaluation:
            if isinstance(self.evaluation_metrics[metric_name], tuple):
                metric_func = self.evaluation_metrics[metric_name][0]
                metric_params = self.evaluation_metrics[metric_name][1]
            else:
                metric_func = self.evaluation_metrics[metric_name]
                metric_params = {}
            metric_val = metric_func(y, y_pred, **metric_params)
            self.evaluation[metric_name].append(metric_val)

            if self.decay_rate:
                if len(self.evaluation_decay[metric_name]) == 0:
                    self.evaluation_decay[metric_name].append(metric_val)
                else:
                    self.evaluation_decay[metric_name].append(
                        self.decay_rate * metric_val + (1 - self.decay_rate) * self.evaluation_decay[metric_name][-1])
            if self.window_size:
                self.evaluation_window[metric_name].append(np.mean(self.evaluation[metric_name][-self.window_size:]))
