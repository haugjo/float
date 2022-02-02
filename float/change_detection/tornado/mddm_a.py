"""McDiarmid Drift Detection Method (Arithmetic Scheme).

Code adopted from https://github.com/alipsgh/tornado, please cite:
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
---
Paper: Pesaranghader, Ali, et al. "McDiarmid Drift Detection Method for Evolving Data Streams."
Published in: International Joint Conference on Neural Network (IJCNN 2018)
URL: https://arxiv.org/abs/1710.02030

Copyright (C) 2022 Johannes Haug

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
from typing import Tuple, List

from float.change_detection.base_change_detector import BaseChangeDetector


class MDDMA(BaseChangeDetector):
    """MDDMA change detector."""
    def __init__(self, n: int = 100, difference: float = 0.01, delta: float = 0.000001, reset_after_drift: bool = False):
        """Inits the change detector.

        Args:
            n: Todo
            difference: Todo
            delta: Todo
            reset_after_drift: See description of base class.
        """
        super().__init__(reset_after_drift=reset_after_drift, error_based=True)

        self._win = []
        self._n = n
        self._difference = difference
        self._delta = delta
        self._e = math.sqrt(0.5 * self._cal_sigma() * (math.log(1 / self._delta, math.e)))
        self._u_max = 0
        self._active_change = False

    def reset(self):
        """Resets the change detector."""
        self._win.clear()
        self._u_max = 0

    def partial_fit(self, pr_scores: List[bool]):
        """Updates the change detector.

        Args:
            pr_scores: Boolean vector indicating correct predictions.
                If True the prediction by the online learner was correct, False otherwise.
        """
        self._active_change = False

        for pr in pr_scores:
            if len(self._win) == self._n:
                self._win.pop(0)
            self._win.append(pr)

            if len(self._win) == self._n:
                u = self._cal_w_sigma()
                self._u_max = u if u > self._u_max else self._u_max
                self._active_change = True if (self._u_max - u > self._e) else False

    def detect_change(self) -> bool:
        """Detects global concept drift."""
        return self._active_change

    def detect_partial_change(self) -> Tuple[bool, list]:
        """Detects partial concept drift.

        Notes:
            MDDMA does not detect partial change.
        """
        return False, []

    def detect_warning_zone(self) -> bool:
        """Detects a warning zone.

        Notes:
            MDDMA does not raise warnings.
        """
        return False

    # ----------------------------------------
    # Tornado Functionality (left unchanged)
    # ----------------------------------------
    def _cal_sigma(self):
        """Tornado-function (left unchanged)."""
        sum_, sigma = 0, 0
        for i in range(self._n):
            sum_ += (1 + i * self._difference)
        for i in range(self._n):
            sigma += math.pow((1 + i * self._difference) / sum_, 2)
        return sigma

    def _cal_w_sigma(self):
        """Tornado-function (left unchanged)."""
        total_sum, win_sum = 0, 0
        for i in range(self._n):
            total_sum += 1 + i * self._difference
            win_sum += self._win[i] * (1 + i * self._difference)
        return win_sum / total_sum
