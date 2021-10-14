"""Exponentially Weigthed Moving Average Drift Detection Method.

Code adopted from https://github.com/alipsgh/tornado, please cite:
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
---
Paper: Ross, Gordon J., et al. "Exponentially weighted moving average charts for detecting concept drift."
Published in: Pattern Recognition Letters 33.2 (2012): 191-198.
URL: https://arxiv.org/pdf/1212.6018.pdf

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


class EWMA(BaseChangeDetector):
    """EWMA change detector."""
    def __init__(self, min_instance: int = 30, lambda_: float = 0.2, reset_after_drift: bool = False):
        """Inits the change detector.

        Args:
            min_instance: Todo
            lambda_: Todo
            reset_after_drift: See description of base class.
        """
        super().__init__(reset_after_drift=reset_after_drift, error_based=True)

        self._MINIMUM_NUM_INSTANCES = min_instance
        self._m_n = 1.0
        self._m_sum = 0.0
        self._m_p = 0.0
        self._m_s = 0.0
        self._z_t = 0.0
        self._lambda_ = lambda_
        self._active_change = False
        self._active_warning = False

    def reset(self):
        """Resets the change detector."""
        self._m_n = 1
        self._m_sum = 0
        self._m_p = 0
        self._m_s = 0
        self._z_t = 0

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
        self._m_sum += pr
        self._m_p = self._m_sum / self._m_n
        self._m_s = math.sqrt(
            self._m_p * (1.0 - self._m_p) * self._lambda_ * (1.0 - math.pow(1.0 - self._lambda_, 2.0 * self._m_n)) / (
                        2.0 - self._lambda_))
        self._m_n += 1

        self._z_t += self._lambda_ * (pr - self._z_t)
        L_t = 3.97 - 6.56 * self._m_p + 48.73 * math.pow(self._m_p, 3) - 330.13 * math.pow(self._m_p, 5) \
              + 848.18 * math.pow(self._m_p, 7)

        # 2. UPDATING WARNING AND DRIFT STATUSES
        if self._m_n < self._MINIMUM_NUM_INSTANCES:
            return

        if self._z_t > self._m_p + L_t * self._m_s:
            self._active_change = True
        elif self._z_t > self._m_p + 0.5 * L_t * self._m_s:
            self._active_warning = True

    def detect_change(self) -> bool:
        """Detects global concept drift."""
        return self._active_change

    def detect_partial_change(self) -> Tuple[bool, list]:
        """Detects partial concept drift.

        Notes:
            EWMA does not detect partial change.
        """
        return False, []

    def detect_warning_zone(self) -> bool:
        """Detects a warning zone."""
        return self._active_warning
