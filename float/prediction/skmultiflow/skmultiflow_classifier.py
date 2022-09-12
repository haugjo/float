"""Scikit-Multiflow Predictive Model Wrapper.

This module contains a wrapper class for scikit-multiflow predictive models.

Copyright (C) 2022 Johannes Haug.
"""
from numpy.typing import ArrayLike
from skmultiflow.core import ClassifierMixin
from typing import Optional

from float.prediction import BasePredictor


class SkmultiflowClassifier(BasePredictor):
    """Wrapper for scikit-multiflow predictive models.

    Attributes:
        model (ClassifierMixin): The scikit-multiflow predictor object.
        classes (list): A list of all unique classes.
    """
    def __init__(self, model: ClassifierMixin, classes: list, reset_after_drift: bool = False):
        """Inits the wrapper.

        Args:
            model: The scikit-multiflow predictor object.
            classes: A list of all unique classes.
            reset_after_drift: A boolean indicating if the predictor will be reset after a drift was detected.
        """
        self.model = model
        self.classes = classes
        super().__init__(reset_after_drift=reset_after_drift)

    def partial_fit(self, X: ArrayLike, y: ArrayLike, sample_weight: Optional[ArrayLike] = None):
        """Updates the predictor.

        Args:
            X: Array/matrix of observations.
            y: Array of corresponding labels.
            sample_weight: Weights per sample. Not used by float at the moment, i.e., all observations in x receive
                equal weight in a pipeline run.
        """
        self.model.partial_fit(X=X, y=y, classes=self.classes, sample_weight=sample_weight)

    def predict(self, X: ArrayLike) -> ArrayLike:
        """Predicts the target values.

        Args:
            X: Array/matrix of observations.

        Returns:
            ArrayLike: Predicted labels for all observations.
        """
        return self.model.predict(X=X)

    def predict_proba(self, X: ArrayLike) -> ArrayLike:
        """Predicts the probability of target values.

        Args:
            X: Array/matrix of observations.

        Returns:
            ArrayLike: Predicted probability per class label for all observations.
        """
        return self.model.predict_proba(X=X)

    def reset(self):
        """Resets the predictor."""
        self.model.reset()
        self.has_been_trained = False
