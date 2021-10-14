"""Hoeffding's Bound based Drift Detection Method (A_test Scheme).

Code adopted from https://github.com/alipsgh/tornado, please cite:
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
---
Paper: Frías-Blanco, Isvani, et al. "Online and non-parametric drift detection methods based on Hoeffding’s bounds."
Published in: IEEE Transactions on Knowledge and Data Engineering 27.3 (2015): 810-823.
URL: http://ieeexplore.ieee.org/abstract/document/6871418/

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
import math
from typing import Tuple

from float.change_detection.base_change_detector import BaseChangeDetector


class HDDMA(BaseChangeDetector):
    """HDDMA change detector."""
    def __init__(self, drift_confidence: float = 0.001, warning_confidence: float = 0.005, test_type: str = 'two-sided',
                 reset_after_drift: bool = False):
        """ Inits the change detector.

        Args:
            drift_confidence: Todo
            warning_confidence: Todo
            test_type: Todo
            reset_after_drift: See description of base class.
        """
        super().__init__(reset_after_drift=reset_after_drift, error_based=True)

        self._drift_confidence = drift_confidence
        self._warning_confidence = warning_confidence
        self._test_type = test_type
        self._n_min = 0
        self._c_min = 0
        self._total_n = 0
        self._total_c = 0
        self._n_max = 0
        self._c_max = 0
        self._active_change = False
        self._active_warning = False

    def reset(self):
        """Resets the change detector."""
        self._n_min = 0
        self._c_min = 0
        self._total_n = 0
        self._total_c = 0
        self._n_max = 0
        self._c_max = 0

    def partial_fit(self, pr: bool):
        """Updates the change detector.

        Args:
            pr: Boolean indicating a correct prediction.
                If True the prediction by the online learner was correct, False otherwise.
        """
        pr = 1 if pr is False else 0

        self._active_change = False
        self._active_warning = False

        # 1. UPDATING STATS
        self._total_n += 1
        self._total_c += pr

        if self._n_min == 0:
            self._n_min = self._total_n
            self._c_min = self._total_c

        if self._n_max == 0:
            self._n_max = self._total_n
            self._c_max = self._total_c

        cota = math.sqrt((1.0 / (2 * self._n_min)) * math.log(1.0 / self._drift_confidence, math.e))
        cota1 = math.sqrt((1.0 / (2 * self._total_n)) * math.log(1.0 / self._drift_confidence, math.e))
        if self._c_min / self._n_min + cota >= self._total_c / self._total_n + cota1:
            self._c_min = self._total_c
            self._n_min = self._total_n

        cota = math.sqrt((1.0 / (2 * self._n_max)) * math.log(1.0 / self._drift_confidence, math.e))
        if self._c_max / self._n_max - cota <= self._total_c / self._total_n - cota1:
            self._c_max = self._total_c
            self._n_max = self._total_n

        if self._mean_incr(self._drift_confidence):
            self._n_min = self._n_max = self._total_n = 0
            self._c_min = self._c_max = self._total_c = 0
            self._active_change = True
        elif self._mean_incr(self._warning_confidence):
            self._active_warning = True

        # 2. UPDATING WARNING AND DRIFT STATUSES
        if self._test_type == 'two-sided' and self._mean_decr():
            self._n_min = self._n_max = self._total_n = 0
            self._c_min = self._c_max = self._total_c = 0

    def detect_change(self) -> bool:
        """Detects global concept drift."""
        return self._active_change

    def detect_partial_change(self) -> Tuple[bool, list]:
        """Detects partial concept drift.

        Notes:
            HDDMA does not detect partial change.
        """
        return False, []

    def detect_warning_zone(self) -> bool:
        """Detects a warning zone."""
        return self._active_warning

    # ----------------------------------------
    # Tornado Functionality (left unchanged)
    # ----------------------------------------
    def _mean_incr(self, confidence_level):
        """Tornado-function (left unchanged)."""
        if self._n_min == self._total_n:
            return False
        m = (self._total_n - self._n_min) / self._n_min * (1.0 / self._total_n)
        cota = math.sqrt((m / 2) * math.log(2.0 / confidence_level, math.e))
        return self._total_c / self._total_n - self._c_min / self._n_min >= cota

    def _mean_decr(self):
        """Tornado-function (left unchanged)."""
        if self._n_max == self._total_n:
            return False
        m = (self._total_n - self._n_max) / self._n_max * (1.0 / self._total_n)
        cota = math.sqrt((m / 2) * math.log(2.0 / self._drift_confidence, math.e))
        return self._c_max / self._n_max - self._total_c / self._total_n >= cota
