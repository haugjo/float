"""Base Online Predictor.

This module encapsulates functionality for online predictive modelling.
The abstract BasePredictor class should be used as super class for all online predictive models.

Copyright (C) 2022 Johannes Haug.
"""
from abc import ABCMeta, abstractmethod
from numpy.typing import ArrayLike
from typing import Optional


class BasePredictor(metaclass=ABCMeta):
    """Abstract base class for online predictive models.

    Attributes:
        reset_after_drift (bool): A boolean indicating if the predictor will be reset after a drift was detected.
        has_been_trained (bool): A boolean indicating if the predictor has been trained at least once.
    """

    def __init__(self, reset_after_drift: bool):
        """Inits the predictor.

        Args:
            reset_after_drift: A boolean indicating if the predictor will be reset after a drift was detected.
        """
        self.reset_after_drift = reset_after_drift
        self.has_been_trained = False

    @abstractmethod
    def partial_fit(self, X: ArrayLike, y: ArrayLike, sample_weight: Optional[ArrayLike] = None):
        """Updates the predictor.

        Args:
            X: Array/matrix of observations.
            y: Array of corresponding labels.
            sample_weight: Weights per sample. If no weights are provided, we weigh observations uniformly.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: ArrayLike) -> ArrayLike:
        """Predicts the target values.

        Args:
            X: Array/matrix of observations.

        Returns:
            ArrayLike: Predicted labels for all observations.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X: ArrayLike) -> ArrayLike:
        """Predicts the probability of target values.

        Args:
            X: Array/matrix of observations.

        Returns:
            ArrayLike: Predicted probability per class label for all observations.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """Resets the predictor."""
        raise NotImplementedError
