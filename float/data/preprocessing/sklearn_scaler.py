"""Sklearn Scaling Function Wrapper.

This module contains a wrapper for the scikit-learn scaling functions.

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
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from typing import Any
import warnings

from float.data.preprocessing import BaseScaler


class SklearnScaler(BaseScaler):
    """Wrapper for sklearn scaler functions.

    Attributes:
        scaler_obj (Any): A scikit-learn scaler object (e.g. MinMaxScaler)
    """
    def __init__(self, scaler_obj: Any, reset_after_drift: bool = False):
        """Inits the sklearn scaler.

        Args:
            scaler_obj: A scikit-learn scaler object (e.g. MinMaxScaler)
            reset_after_drift: See description in base class.
        """
        super().__init__(reset_after_drift=reset_after_drift)

        self.scaler_obj = scaler_obj
        self._has_partial_fit = True
        self._must_be_fitted = False
        self._validate()

    def partial_fit(self, X: ArrayLike):
        """Updates the scaler."""
        if self._must_be_fitted:
            self.scaler_obj.fit(X)
            self._must_be_fitted = False
        elif self._has_partial_fit:
            self.scaler_obj.partial_fit(X)

    def transform(self, X: ArrayLike) -> ArrayLike:
        """Scales the given observations."""
        return self.scaler_obj.transform(X)

    def reset(self):
        """Resets the scaler.

        We automatically re-fit the scaler upon the next call to partial_fit.
        """
        self._must_be_fitted = True

    def _validate(self):
        """Validates the provided scaler object.

        Raises:
            TypeError: If scaler object does neither have a partial_fit nor a fit function.
            TypeError: If scaler object does not have a transform function.
        """
        # Check if scaler object has a partial fit function
        partial_fit_func = getattr(self.scaler_obj, "partial_fit", None)
        if not callable(partial_fit_func):
            # Check if scaler object has a fit function
            fit_func = getattr(self.scaler_obj, "fit", None)
            if not callable(fit_func):
                raise TypeError("{} is not a valid sklearn scaler (missing 'fit' or 'partial_fit' function).".format(
                    type(self.scaler_obj).__name__))
            else:
                try:
                    self._has_partial_fit = False
                    warnings.warn(
                        "The {} scaler has no partial_fit function and will thus not be updated, which may mitigate "
                        "the overall performance.".format(type(self.scaler_obj).__name__))
                    check_is_fitted(self.scaler_obj)  # Check if scaler object has already been fitted
                except NotFittedError:
                    self._must_be_fitted = True
                    warnings.warn('Sklearn scaler object {} has not been fitted and will be fitted on the first batch '
                                  'of observations.'.format(type(self.scaler_obj).__name__))
                    pass

        # Check if scaler object has a transform function
        transform_func = getattr(self.scaler_obj, "transform", None)
        if not callable(transform_func):
            raise TypeError("{} is not a valid sklearn scaler (missing 'transform' function).".format(
                type(self.scaler_obj).__name__))
