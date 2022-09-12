"""River Predictive Model Wrapper.

This module contains a wrapper class for river predictive models.

Copyright (C) 2022 Johannes Haug.
"""
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from river.base import Classifier
from typing import Optional, List

from float.prediction import BasePredictor


class RiverClassifier(BasePredictor):
    """Wrapper for river predictive models.

    Attributes:
        model (ClassifierMixin): The river predictor object.
        feature_names (List[str]): A list of all feature names.
    """
    def __init__(self, model: Classifier, feature_names: List[str], reset_after_drift: bool = False):
        """Inits the wrapper.

        Args:
            model: The river predictor object.
            feature_names: A list of all feature names.
            reset_after_drift: A boolean indicating if the predictor will be reset after a drift was detected.
        """
        self.init_model = model.clone()
        self.model = model
        self.feature_names = feature_names

        if isinstance(self.model, Classifier):
            if getattr(self.model, 'learn_many', None):
                self.can_mini_batch = True
            else:
                self.can_mini_batch = False
        else:
            raise TypeError('River classifier class {} is not supported.'.format(type(self.model)))

        super().__init__(reset_after_drift=reset_after_drift)

    def partial_fit(self, X: ArrayLike, y: ArrayLike, sample_weight: Optional[ArrayLike] = 1):
        """Updates the predictor.

        Args:
            X: Array/matrix of observations.
            y: Array of corresponding labels.
            sample_weight: Weights per sample. Not used by float at the moment, i.e., all observations in x receive
                equal weight in a pipeline run.
        """
        if self.can_mini_batch:
            X = pd.DataFrame(X, columns=self.feature_names)
            y = pd.Series(y)
            self.model.learn_many(X=X, y=y, w=sample_weight)
        else:
            x = {key: value[0] for key, value in zip(self.feature_names, X.reshape((-1, 1)))}
            self.model.learn_one(x=x, y=bool(y), w=sample_weight)

    def predict(self, X: ArrayLike) -> ArrayLike:
        """Predicts the target values.

        Args:
            X: Array/matrix of observations.

        Returns:
            ArrayLike: Predicted labels for all observations.
        """
        if self.can_mini_batch:
            X = pd.DataFrame(X, columns=self.feature_names)
            return self.model.predict_many(X=X)
        else:
            x = {key: value[0] for key, value in zip(self.feature_names, X.reshape((-1, 1)))}
            return np.array([self.model.predict_one(x=x)])

    def predict_proba(self, X: ArrayLike) -> ArrayLike:
        """Predicts the probability of target values.

        Args:
            X: Array/matrix of observations.

        Returns:
            ArrayLike: Predicted probability per class label for all observations.
        """
        if self.can_mini_batch:
            X = pd.DataFrame(X, columns=self.feature_names)
            return self.model.predict_proba_many(X=X)
        else:
            x = {key: value[0] for key, value in zip(self.feature_names, X.reshape((-1, 1)))}
            return np.array([self.model.predict_proba_one(x=x)])

    def reset(self):
        """Resets the predictor."""
        self.model = self.init_model.clone()
        self.has_been_trained = False
