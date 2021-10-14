"""Reactive Drift Detection Method.

Code adopted from https://github.com/alipsgh/tornado, please cite:
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
---
Paper: Barros, Roberto, et al. "RDDM: Reactive drift detection method."
Published in: Expert Systems with Applications. Elsevier, 2017.
URL: https://www.sciencedirect.com/science/article/pii/S0957417417305614

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
import sys
from typing import Tuple

from float.change_detection.base_change_detector import BaseChangeDetector


class RDDM(BaseChangeDetector):
    """RDDM change detector."""
    def __init__(self, min_instance: int = 129, warning_level: float = 1.773, drift_level: float = 2.258,
                 max_size_concept: int = 40000, min_size_stable_concept: int = 7000, warn_limit: int = 1400,
                 reset_after_drift: bool = False):
        """Inits the change detector.

        Args:
            min_instance: Todo
            warning_level: Todo
            drift_level: Todo
            max_size_concept: Todo
            min_size_stable_concept: Todo
            warn_limit: Todo
            reset_after_drift: See description of base class.
        """
        super().__init__(reset_after_drift=reset_after_drift, error_based=True)

        self._min_num_instance = min_instance
        self._warning_level = warning_level
        self._drift_level = drift_level
        self._max_concept_size = max_size_concept
        self._min_size_stable_concept = min_size_stable_concept
        self._warn_limit = warn_limit

        self._m_n = 1
        self._m_p = 1
        self._m_s = 0
        self._m_p_min = sys.maxsize
        self._m_s_min = sys.maxsize
        self._m_p_s_min = sys.maxsize

        self._stored_predictions = [0 for _ in range(self._min_size_stable_concept)]
        self._num_stored_instances = 0
        self._first_pos = 0
        self._last_pos = -1
        self._last_warn_pos = -1
        self._last_warn_inst = -1
        self._inst_num = 0
        self._rddm_drift = False
        self._is_change_detected = False
        self._is_warning_zone = False

        self._active_change = False
        self._active_warning = False

    def reset(self):
        """Resets the change detector."""
        self._m_n = 1
        self._m_p = 1
        self._m_s = 0
        self._m_p_min = sys.maxsize
        self._m_s_min = sys.maxsize
        self._m_p_s_min = sys.maxsize

    def partial_fit(self, pr):
        """Updates the change detector.

        Args:
            pr: Boolean indicating a correct prediction.
                If True the prediction by the online learner was correct, False otherwise.
        """
        pr = 1 if pr is False else 0

        self._active_change = False
        self._active_warning = False

        if self._rddm_drift:  #
            self._reset_rddm()  #
            if self._last_warn_pos != -1:  #
                self._first_pos = self._last_warn_pos  #
                self._num_stored_instances = self._last_pos - self._first_pos + 1  #
                if self._num_stored_instances <= 0:  #
                    self._num_stored_instances += self._min_size_stable_concept  #

            pos = self._first_pos  #
            for i in range(0, self._num_stored_instances):  #
                self._m_p += ((self._stored_predictions[pos] - self._m_p) / self._m_n)  #
                self._m_s = math.sqrt(self._m_p * (1 - self._m_p) / self._m_n)
                if self._is_change_detected and (self._m_n > self._min_num_instance) and (
                        self._m_p + self._m_s < self._m_p_s_min):
                    self._m_p_min = self._m_p
                    self._m_s_min = self._m_s
                    self._m_p_s_min = self._m_p + self._m_s
                self._m_n += 1
                pos = (pos + 1) % self._min_size_stable_concept

            self._last_warn_pos = -1
            self._last_warn_inst = -1
            self._rddm_drift = False
            self._is_change_detected = False

        self._last_pos = (self._last_pos + 1) % self._min_size_stable_concept
        self._stored_predictions[self._last_pos] = pr
        if self._num_stored_instances < self._min_size_stable_concept:
            self._num_stored_instances += 1
        else:
            self._first_pos = (self._first_pos + 1) % self._min_size_stable_concept
            if self._last_warn_pos == self._last_pos:
                self._last_warn_pos = -1

        self._m_p += (pr - self._m_p) / self._m_n
        self._m_s = math.sqrt(self._m_p * (1 - self._m_p) / self._m_n)

        self._inst_num += 1
        self._m_n += 1
        self._is_warning_zone = False

        if self._m_n <= self._min_num_instance:
            return

        if self._m_p + self._m_s < self._m_p_s_min:
            self._m_p_min = self._m_p
            self._m_s_min = self._m_s
            self._m_p_s_min = self._m_p + self._m_s

        if self._m_p + self._m_s > self._m_p_min + self._drift_level * self._m_s_min:
            self._is_change_detected, self._active_change = True, True
            self._rddm_drift = True
            if self._last_warn_inst == -1:
                self._first_pos = self._last_pos
                self._num_stored_instances = 1
            return

        if self._m_p + self._m_s > self._m_p_min + self._warning_level * self._m_s_min:
            if (self._last_warn_inst != -1) and (self._last_warn_inst + self._warn_limit <= self._inst_num):
                self._is_change_detected, self._active_change = True, True
                self._rddm_drift = True
                self._first_pos = self._last_pos
                self._num_stored_instances = 1
                self._last_warn_pos = -1
                self._last_warn_inst = -1
                return

            self._is_warning_zone, self._active_warning = True, True
            if self._last_warn_inst == -1:
                self._last_warn_inst = self._inst_num
                self._last_warn_pos = self._last_pos
        else:
            self._last_warn_inst = -1
            self._last_warn_pos = -1

        if self._m_n > self._max_concept_size and self._is_warning_zone is False:
            self._rddm_drift = True

    def detect_change(self) -> bool:
        """Detects global concept drift."""
        return self._active_change

    def detect_partial_change(self) -> Tuple[bool, list]:
        """Detects partial concept drift.

        Notes:
            RDDM does not detect partial change.
        """
        return False, []

    def detect_warning_zone(self) -> bool:
        """Detects a warning zone."""
        return self._active_warning

    # ----------------------------------------
    # Tornado Functionality (left unchanged)
    # ----------------------------------------
    def _reset_rddm(self):
        """Tornado-function (left unchanged)."""
        self._m_n = 1
        self._m_p = 1
        self._m_s = 0
        if self._is_change_detected:
            self._m_p_min = sys.maxsize
            self._m_s_min = sys.maxsize
            self._m_p_s_min = sys.maxsize
