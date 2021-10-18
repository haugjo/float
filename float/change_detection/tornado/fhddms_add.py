"""Additive Stacking Fast Hoeffding Drift Detection Method.

Code adopted from https://github.com/alipsgh/tornado, please cite:
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
---
Paper: Reservoir of Diverse Adaptive Learners and Stacking Fast Hoeffding Drift Detection Methods for Evolving Data Streams
URL: https://arxiv.org/pdf/1709.02457.pdf

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


class FHDDMSAdd(BaseChangeDetector):
    """FHDDSMAdd change detector."""
    def __init__(self, m: int = 4, n: int = 25, delta: float = 0.000001, reset_after_drift: bool = False):
        """ Initialize the concept drift detector

        Args:
            m: Todo
            n: Todo
            delta: Todo
            reset_after_drift: See description of base class.
        """
        super().__init__(reset_after_drift=reset_after_drift, error_based=True)

        self._ELEMENT_SIZE = n
        self._DELTA = delta
        self._stack = []
        self._init_stack(size=m)
        self._first_round = True
        self._counter = 0
        self._mu_max_short = 0
        self._mu_max_large = 0
        self._num_ones = 0
        self._active_change = False

    def reset(self):
        """Resets the change detector."""
        self._init_stack(size=len(self._stack))
        self._first_round = True
        self._counter = 0
        self._mu_max_short = 0
        self._mu_max_large = 0
        self._num_ones = 0

    def partial_fit(self, pr):
        """Updates the change detector.

        Args:
            pr: Boolean indicating a correct prediction.
                If True the prediction by the online learner was correct, False otherwise.
        """
        self._active_change = False

        self._counter += 1

        if self._counter == (len(self._stack) * self._ELEMENT_SIZE) + 1:
            self._counter -= self._ELEMENT_SIZE
            self._num_ones -= self._stack[0]
            self._stack.pop(0)
            self._stack.append(0.0)
            if self._first_round is True:
                self._first_round = False

        if self._first_round is True:
            index = int(self._counter / self._ELEMENT_SIZE)
            if index == len(self._stack):
                index -= 1
        else:
            index = len(self._stack) - 1

        if pr is True:
            self._stack[index] += 1
            self._num_ones += 1

        # TESTING THE NEW SUB-WINDOWS
        if self._counter % self._ELEMENT_SIZE == 0:
            m_temp = self._stack[index] / self._ELEMENT_SIZE
            if self._mu_max_short < m_temp:
                self._mu_max_short = m_temp
            if self._mu_max_short - m_temp > self.__cal_hoeffding_bound(n=self._ELEMENT_SIZE):
                self._active_change = True

        # TESTING THE WHOLE WINDOW
        if self._counter == len(self._stack) * self._ELEMENT_SIZE:
            m_temp = self._num_ones / (len(self._stack) * self._ELEMENT_SIZE)
            if self._mu_max_large < m_temp:
                self._mu_max_large = m_temp
            if self._mu_max_large - m_temp > self.__cal_hoeffding_bound(n=len(self._stack) * self._ELEMENT_SIZE):
                self._active_change = True

    def detect_change(self) -> bool:
        """Detects global concept drift."""
        return self._active_change

    def detect_partial_change(self) -> Tuple[bool, list]:
        """Detects partial concept drift.

        Notes:
            FHDDMSAdd does not detect partial change.
        """
        return False, []

    def detect_warning_zone(self) -> bool:
        """Detects a warning zone.

        Notes:
            FHDDMSAdd does not raise warnings.
        """
        return False

    # ----------------------------------------
    # Tornado Functionality (left unchanged)
    # ----------------------------------------
    def _init_stack(self, size):
        """Tornado-function (left unchanged)."""
        self._stack.clear()
        for i in range(0, size):
            self._stack.append(0.0)

    def __cal_hoeffding_bound(self, n):
        """Tornado-function (left unchanged)."""
        return math.sqrt(math.log((1 / self._DELTA), math.e) / (2 * n))
