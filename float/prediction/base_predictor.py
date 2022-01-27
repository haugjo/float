"""Base Online Predictor Module.

This module encapsulates functionality for online predictive modelling.
The abstract BaseFeatureSelector class should be used as super class for all online predictive models.

Copyright (C) 2021 Johannes Haug

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
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
        """Predicts the arget values.

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
