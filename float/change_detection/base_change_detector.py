"""Base Change Detection Module.

This module encapsulates functionality for global and partial (i.e. feature-wise) concept drift detection.
The abstract BaseChangeDetector class should be used as super class for all concept drift detection methods.

Copyright (C) 2022 Johannes Haug.
"""
from abc import ABCMeta, abstractmethod
from typing import Tuple, List


class BaseChangeDetector(metaclass=ABCMeta):
    """Abstract base class for change detection models.

    Attributes:
        reset_after_drift (bool): A boolean indicating if the change detector will be reset after a drift was detected.
        error_based (bool):
            A boolean indicating if the change detector partial_fit function requires error measurements (i.e., a
            boolean vector indicating correct predictions) as input from the pipeline. This is true for most change
            detectors. If the attribute is False, the partial_fit function will receive raw input observations and
            targets from the pipeline (e.g., required by ERICS).
        drifts (list): A list of time steps corresponding to detected concept drifts.
        partial_drifts (List[tuple]):
            A list of time steps and features corresponding to detected partial concept drifts. A partial drift is a
            concept drift that is restricted to one or multiple (but not all) input features. Some change detectors are
            able to detect partial concept drift. This attribute is a list of tuples of the form (time step,
            [features under change]).
        warnings (list):
            A list of time steps corresponding to warnings. Some change detectors are able to issue warnings before
            an actual drift alert.
    """
    def __init__(self, reset_after_drift: bool, error_based: bool):
        """Inits the change detector.

        Args:
            reset_after_drift: A boolean indicating if the change detector will be reset after a drift was detected.
            error_based:
                A boolean indicating if the change detector partial_fit function requires error measurements (i.e., a
                boolean vector indicating correct predictions) as input from the pipeline. This is true for most change
                detectors. If the attribute is False, the partial_fit function will receive raw input observations and
                targets from the pipeline (e.g., required by ERICS).
        """
        self.reset_after_drift = reset_after_drift
        self.error_based = error_based
        self.drifts = []
        self.partial_drifts = []
        self.warnings = []

    @abstractmethod
    def reset(self):
        """Resets the change detector."""
        raise NotImplementedError

    @abstractmethod
    def partial_fit(self, *args, **kwargs):
        """Updates the change detector."""
        raise NotImplementedError

    @abstractmethod
    def detect_change(self) -> bool:
        """Detects global concept drift.

        Returns:
            bool: True, if a concept drift was detected, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def detect_partial_change(self) -> Tuple[bool, list]:
        """Detects partial concept drift.

        Returns:
            bool: True, if at least one partial concept drift was detected, False otherwise.
            list: Indices (i.e. relative positions in the feature vector) of input features with detected partial drift.
        """
        raise NotImplementedError

    @abstractmethod
    def detect_warning_zone(self) -> bool:
        """Detects a warning zone.

        Some change detectors issue warnings before the actual drift alert.

        Returns:
            bool: True, if the change detector has detected a warning zone, False otherwise.
        """
        raise NotImplementedError
