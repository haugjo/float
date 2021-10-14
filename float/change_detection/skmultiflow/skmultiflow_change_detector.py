"""Scikit-Multiflow Drift Detection Model Wrapper.

This module contains a wrapper for the scikit-multiflow concept drift detection methods.

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
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
from skmultiflow.drift_detection import ADWIN, DDM, EDDM, HDDM_A, HDDM_W, KSWIN, PageHinkley
from typing import Tuple, Any

from float.change_detection.base_change_detector import BaseChangeDetector


class SkmultiflowChangeDetector(BaseChangeDetector):
    """Wrapper for scikit-multiflow drift detection methods.

    Attributes:
        detector (BaseDriftDetector): The scikit-multiflow concept drift detector object
    """
    def __init__(self, detector: BaseDriftDetector, reset_after_drift: bool = False):
        """Inits the scikit-multiflow change detector.

        Args:
            detector: The scikit-multiflow concept drift detector object
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
        self.detector.add_element(input_value)

    def detect_change(self) -> bool:
        """Detects global concept drift."""
        return self.detector.detected_change()

    def detect_partial_change(self) -> Tuple[bool, list]:
        """Detects partial concept drift.

        Notes:
            Scikit-multiflow change detectors do not detect partial change.
        """
        return False, []

    def detect_warning_zone(self) -> bool:
        """Detects a warning zone."""
        return self.detector.in_warning_zone

    def _validate(self) -> bool:
        """Validate the provided scikit-multiflow drift detector object.

        Returns:
            bool: Boolean indicating whether the method requires error measurements from a predictor.

        Raises:
            TypeError: If the provided detector is not a valid scikit-multiflow drift detection method.
        """
        if isinstance(self.detector, (ADWIN, DDM, EDDM, HDDM_A, HDDM_W, PageHinkley)):
            return True
        elif isinstance(self.detector, KSWIN):
            return False
        else:
            raise TypeError("Scikit-Multiflow Drift Detector Class {} is not supported.".format(type(self.detector)))
