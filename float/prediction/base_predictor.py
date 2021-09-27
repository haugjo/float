from abc import ABCMeta, abstractmethod
import numpy as np


class BasePredictor(metaclass=ABCMeta):
    """
    Abstract base class which serves as both a wrapper for skmultiflow predictive models and a constructor for
    user-defined predictive models.

    Attributes:
        reset_after_drift (bool): indicates whether to reset the predictor after a drift was detected
    """

    def __init__(self, reset_after_drift):
        """
        Initializes the predictor.

        Args:
            reset_after_drift (bool): indicates whether to reset the predictor after a drift was detected
        """
        self.reset_after_drift = reset_after_drift

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

    @abstractmethod
    def reset(self, X, y):
        """
        Reset the predictor.

        Args:
            X (np.ndarray): data samples to train the model with
            y (np.ndarray): target values for all samples in X
        """
        raise NotImplementedError
