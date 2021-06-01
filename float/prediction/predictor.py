from abc import ABCMeta, abstractmethod
import numpy as np


class Predictor(metaclass=ABCMeta):
    """
    Abstract base class which serves as both a wrapper for skmultiflow predictive models and a constructor for
    user-defined predictive models.

    Attributes:
        predictions (list): predicted labels per time step
        testing_times (list): testing times per time step
        training_times (list): training times per time step
    """
    @abstractmethod
    def __init__(self):
        """
        Initializes the predictor.
        """
        self.predictions = []
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

    def evaluate(self):
        # TODO
        raise NotImplementedError

    @abstractmethod
    def _score(self, X, y):
        """
        Returns the accuracy based on the given test samples and true values.

        Args:
            X (np.ndarray): test data samples
            y (np.ndarray): true values for all samples in X

        Returns:
            float: accuracy based on test data and target values
        """
        raise NotImplementedError
