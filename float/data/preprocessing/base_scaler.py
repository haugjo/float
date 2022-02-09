"""Base Scaling Module.

This module encapsulates functionality to scale, i.e. normalize, streaming observations. The abstract BaseScaler should
be used to implement custom scaling methods. A scaler object can be provided to the data loader object.

Copyright (C) 2022 Johannes Haug.
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
    def partial_fit(self, X: ArrayLike):
        """Updates the scaler.

        Args:
            X: Array/matrix of observations.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: ArrayLike) -> ArrayLike:
        """Scales the given observations.

        Args:
            X: Array/matrix of observations.

        Returns:
            ArrayLike: The scaled observations.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """Resets the scaler."""
        raise NotImplementedError
