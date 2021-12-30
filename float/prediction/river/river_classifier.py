"""River Predictive Model Wrapper.

This module contains a wrapper for the river predictive models.

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
        """Inits the river predictor.

        Args:
            model: The river predictor object.
            feature_names: A list of all feature names.
            reset_after_drift: A boolean indicating if the predictor will be reset after a drift was detected.
        """
        self.init_model = model.clone()
        self.model = model
        self.feature_names = feature_names

        self.can_mini_batch = False
        self._validate()
        super().__init__(reset_after_drift=reset_after_drift)

    def partial_fit(self, X: ArrayLike, y: ArrayLike, sample_weight: Optional[ArrayLike] = None):
        """Updates the predictor."""
        if self.can_mini_batch:
            X = pd.DataFrame(X, columns=self.feature_names)
            y = pd.Series(y)
            self.model.learn_many(X=X, y=y)
        else:
            x = {key: value[0] for key, value in zip(self.feature_names, X.reshape((-1, 1)))}
            self.model.learn_one(x=x, y=bool(y))

    def predict(self, X: ArrayLike) -> ArrayLike:
        """Predicts the target values."""
        if self.can_mini_batch:
            X = pd.DataFrame(X, columns=self.feature_names)
            return self.model.predict_many(X=X)
        else:
            x = {key: value[0] for key, value in zip(self.feature_names, X.reshape((-1, 1)))}
            return np.array([self.model.predict_one(x=x)])

    def predict_proba(self, X: ArrayLike) -> ArrayLike:
        """Predicts the probability of target values."""
        if self.can_mini_batch:
            X = pd.DataFrame(X, columns=self.feature_names)
            return self.model.predict_proba_many(X=X)
        else:
            x = {key: value[0] for key, value in zip(self.feature_names, X.reshape((-1, 1)))}
            return np.array([self.model.predict_proba_one(x=x)])

    def reset(self, X: ArrayLike, y: ArrayLike):
        """Resets the predictor and fits to given sample."""
        self.__init__(self.init_model, self.feature_names, self.reset_after_drift)
        self.partial_fit(X=X, y=y)

    def _validate(self):
        """Validate the provided river classifier object.

        Raises:
            TypeError: If the provided classifier is not a valid river classification method.
        """
        if isinstance(self.model, Classifier):
            if getattr(self.model, 'learn_many', None):
                self.can_mini_batch = True
        else:
            raise TypeError('River classifier class {} is not supported.'.format(type(self.model)))

    def clone(self):
        """Returns a clone of this predictor with the weights reset."""
        return type(self)(self.init_model, self.feature_names, self.reset_after_drift)
