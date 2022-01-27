"""River Drift Detection Model Wrapper.

This module contains a wrapper for the river concept drift detection methods.

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
from river.base import DriftDetector
from river.drift import ADWIN, DDM, EDDM, HDDM_A, HDDM_W, PageHinkley
from typing import Tuple, List

from float.change_detection import BaseChangeDetector


class RiverChangeDetector(BaseChangeDetector):
    """Wrapper for river drift detection methods.

    Attributes:
        detector (BaseDriftDetector): The river concept drift detector object
    """
    def __init__(self, detector: DriftDetector, reset_after_drift: bool = False):
        """Inits the river change detector.

        Args:
            detector: The river concept drift detector object
            reset_after_drift: See description of base class.
        """
        self.detector = detector
        self._validate()
        super().__init__(reset_after_drift=reset_after_drift, error_based=True)

    def reset(self):
        """Resets the change detector."""
        self.detector.reset()

    def partial_fit(self, pr_scores: List[bool]):
        """Updates the parameters of the concept drift detection model.

        Args:
            pr_scores: Boolean vector indicating correct predictions.
                If True the prediction by the online learner was correct, False otherwise.
        """
        for pr in pr_scores:
            self.detector.update(pr)

    def detect_change(self) -> bool:
        """Detects global concept drift."""
        return self.detector.change_detected

    def detect_partial_change(self) -> Tuple[bool, list]:
        """Detects partial concept drift.

        Notes:
            River change detectors do not detect partial change.
        """
        return False, []

    def detect_warning_zone(self) -> bool:
        """Detects a warning zone."""
        return self.detector.warning_detected

    def _validate(self):
        """Validates the provided river drift detector object.

        Raises:
            TypeError: If the provided detector is not a valid river drift detection method.
        """
        if not isinstance(self.detector, (ADWIN, DDM, EDDM, HDDM_A, HDDM_W, PageHinkley)):
            raise TypeError("River drift detector class {} is not supported.".format(type(self.detector)))
