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
from river.drift import ADWIN, DDM, EDDM, HDDM_A, HDDM_W, PageHinkley, KSWIN
from typing import Tuple, Any

from float.change_detection.base_change_detector import BaseChangeDetector


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
        error_based = self._validate()
        super().__init__(reset_after_drift=reset_after_drift, error_based=error_based)

    def reset(self):
        """Resets the change detector."""
        self.detector.reset()

    def partial_fit(self, input_value: Any):
        """
        Update the parameters of the concept drift detection model.

        Args:
            input_value: Whatever input value the concept drift detector takes.
        """
        self.detector.update(input_value)

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

    def _validate(self) -> bool:
        """Validate the provided river drift detector object.

        Returns:
            bool: Boolean indicating whether the method requires error measurements from a predictor.

        Raises:
            TypeError: If the provided detector is not a valid river drift detection method.
        """
        if isinstance(self.detector, (ADWIN, DDM, EDDM, HDDM_A, HDDM_W, PageHinkley)):
            return True
        elif isinstance(self.detector, KSWIN):  # TODO also the case for river?
            return False
        else:
            raise TypeError("River drift detector class {} is not supported.".format(type(self.detector)))
