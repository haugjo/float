"""Base Scaling Module.

This module encapsulates functionality to scale, i.e. normalize, streaming observations. The abstract BaseScaler should
be used to implement custom scaling methods. A scaler object can be provided to the data loader object.

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


class BaseScaler(metaclass=ABCMeta):
    """Abstract Base Class for online data scaling.

    Attributes:
        reset_after_drift (bool): A boolean indicating if the scaler will be reset after a drift was detected.
    """
    def __init__(self, reset_after_drift: bool):
        """Initializes the data scaler.

        Args:
            reset_after_drift: A boolean indicating if the scaler will be reset after a drift was detected.
        """
        self.reset_after_drift = reset_after_drift

    @abstractmethod
    def partial_fit(self, x: ArrayLike):
        """Updates the scaler.

        Args:
            x: Array/matrix of observations.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, x: ArrayLike) -> ArrayLike:
        """Scales the given observations.

        Args:
            x: Array/matrix of observations.

        Returns:
            ArrayLike: The scaled observations.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """Resets the scaler."""
        raise NotImplementedError
