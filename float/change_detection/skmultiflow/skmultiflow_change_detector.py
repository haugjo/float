"""Scikit-Multiflow Change Detection Model Wrapper.

This module contains a wrapper class for scikit-multiflow concept drift detection methods.

Copyright (C) 2022 Johannes Haug.
"""
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
from skmultiflow.drift_detection import ADWIN, DDM, EDDM, HDDM_A, HDDM_W, PageHinkley
from typing import Tuple, List

from float.change_detection import BaseChangeDetector


class SkmultiflowChangeDetector(BaseChangeDetector):
    """Wrapper class for scikit-multiflow change detection classes.

    Attributes:
        detector (BaseDriftDetector): The scikit-multiflow concept drift detector object.
    """
    def __init__(self, detector: BaseDriftDetector, reset_after_drift: bool = False):
        """Inits the wrapper.

        Args:
            detector: The scikit-multiflow concept drift detector object.
            reset_after_drift: A boolean indicating if the change detector will be reset after a drift was detected.
        """
        self.detector = detector
        self._validate()
        super().__init__(reset_after_drift=reset_after_drift, error_based=True)

    def reset(self):
        """Resets the change detector."""
        self.detector.reset()

    def partial_fit(self, pr_scores: List[bool]):
        """Updates the change detector.

        Args:
            pr_scores:
                A boolean vector indicating correct predictions. 'True' values indicate that the prediction by the
                online learner was correct, otherwise the vector contains 'False'.
        """
        for pr in pr_scores:
            self.detector.add_element(pr)

    def detect_change(self) -> bool:
        """Detects global concept drift.

        Returns:
            bool: True, if a concept drift was detected, False otherwise.
        """
        return self.detector.detected_change()

    def detect_partial_change(self) -> Tuple[bool, list]:
        """Detects partial concept drift.

        Notes:
            Scikit-multiflow change detectors do not detect partial change.
        """
        return False, []

    def detect_warning_zone(self) -> bool:
        """Detects a warning zone.

        Returns:
            bool: True, if the change detector has detected a warning zone, False otherwise.
        """
        return self.detector.detected_warning_zone()

    def _validate(self):
        """Validates the provided scikit-multiflow drift detector object.

        Raises:
            TypeError: If the provided detector is not a valid scikit-multiflow drift detection object.
        """
        if not isinstance(self.detector, (ADWIN, DDM, EDDM, HDDM_A, HDDM_W, PageHinkley)):
            raise TypeError("Scikit-multiflow drift detector class {} is not supported.".format(type(self.detector)))
