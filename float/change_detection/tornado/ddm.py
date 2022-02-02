"""Drift Detection Method.

Code adopted from https://github.com/alipsgh/tornado, please cite:
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
---
Paper: Gama, Joao, et al. "Learning with drift detection."
Published in: Brazilian Symposium on Artificial Intelligence. Springer, Berlin, Heidelberg, 2004.
URL: https://link.springer.com/chapter/10.1007/978-3-540-28645-5_29

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
import sys
from typing import Tuple, List

from float.change_detection.base_change_detector import BaseChangeDetector


class DDM(BaseChangeDetector):
    """DDM change detector."""
    def __init__(self, min_instance: int = 30, reset_after_drift: bool = False):
        """Inits the change detector.

        Args:
            min_instance: Todo
            reset_after_drift: See description of base class.
        """
        super().__init__(reset_after_drift=reset_after_drift, error_based=True)

        self._MINIMUM_NUM_INSTANCES = min_instance
        self._NUM_INSTANCES_SEEN = 1
        self.__P = 1
        self.__S = 0
        self.__P_min = sys.maxsize
        self.__S_min = sys.maxsize
        self._active_change = False
        self._active_warning = False

    def reset(self):
        """Resets the change detector."""
        self._NUM_INSTANCES_SEEN = 1
        self.__P = 1
        self.__S = 0
        self.__P_min = sys.maxsize
        self.__S_min = sys.maxsize

    def partial_fit(self, pr_scores: List[bool]):
        """Updates the change detector.

        Args:
            pr_scores: Boolean vector indicating correct predictions.
                If True the prediction by the online learner was correct, False otherwise.
        """
        self._active_change = False
        self._active_warning = False

        for pr in pr_scores:
            pr = 1 if pr is False else 0

            # 1. UPDATING STATS
            self.__P += (pr - self.__P) / self._NUM_INSTANCES_SEEN
            self.__S = math.sqrt(self.__P * (1 - self.__P) / self._NUM_INSTANCES_SEEN)

            self._NUM_INSTANCES_SEEN += 1

            if self._NUM_INSTANCES_SEEN < self._MINIMUM_NUM_INSTANCES:
                return

            if self.__P + self.__S <= self.__P_min + self.__S_min:
                self.__P_min = self.__P
                self.__S_min = self.__S

            # 2. UPDATING WARNING AND DRIFT STATUSES
            current_level = self.__P + self.__S
            warning_level = self.__P_min + 2 * self.__S_min
            drift_level = self.__P_min + 3 * self.__S_min

            if current_level > warning_level:
                self._active_warning = True

            if current_level > drift_level:
                self._active_change = True

    def detect_change(self) -> bool:
        """Detects global concept drift."""
        return self._active_change

    def detect_partial_change(self) -> Tuple[bool, list]:
        """Detects partial concept drift.

        Notes:
            DDM does not detect partial change.
        """
        return False, []

    def detect_warning_zone(self) -> bool:
        """Detects a warning zone."""
        return self._active_warning
