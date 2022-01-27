"""Scikit-Multiflow Predictive Model Wrapper.

This module contains a wrapper for the scikit-multiflow predictive models.

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
        """Inits the scikit-multiflow predictor.

        Args:
            model: The scikit-multiflow predictor object.
            classes: A list of all unique classes.
            reset_after_drift: A boolean indicating if the predictor will be reset after a drift was detected.
        """
        # self.init_model = model.copy()
        self.model = model
        self.classes = classes
        super().__init__(reset_after_drift=reset_after_drift)

    def partial_fit(self, X: ArrayLike, y: ArrayLike, sample_weight: Optional[ArrayLike] = None):
        """Updates the predictor."""
        self.model.partial_fit(X=X, y=y, classes=self.classes, sample_weight=sample_weight)

    def predict(self, X: ArrayLike) -> ArrayLike:
        """Predicts the target values."""
        return self.model.predict(X=X)

    def predict_proba(self, X: ArrayLike) -> ArrayLike:
        """Predicts the probability of target values."""
        return self.model.predict_proba(X=X)

    def reset(self):
        """Resets the predictor."""
        self.model.reset()
        self.has_been_trained = False
