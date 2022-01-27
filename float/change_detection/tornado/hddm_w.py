"""Hoeffding's Bound based Drift Detection Method (W_test Scheme).

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
import sys
from typing import Tuple, List

from float.change_detection.base_change_detector import BaseChangeDetector


class HDDMW(BaseChangeDetector):
    """ HDDMW change detector."""
    def __init__(self,
                 drift_confidence: float = 0.001,
                 warning_confidence: float = 0.005,
                 lambda_: float = 0.05,
                 test_type: str = 'one-sided',
                 reset_after_drift: bool = False):
        """ Inits the change detector.

        Args:
            drift_confidence: Todo
            warning_confidence: Todo
            lambda_: Todo
            test_type: Todo
            reset_after_drift: See description of base class.
        """
        super().__init__(reset_after_drift=reset_after_drift, error_based=True)

        self._total = _SampleInfo()
        self._sample1_decr_monitoring = _SampleInfo()
        self._sample1_incr_monitoring = _SampleInfo()
        self._sample2_decr_monitoring = _SampleInfo()
        self._sample2_incr_monitoring = _SampleInfo()
        self._incr_cut_point = sys.float_info.max
        self._decr_cut_point = sys.float_info.min
        self._drift_confidence = drift_confidence
        self._warning_confidence = warning_confidence
        self._lambda_ = lambda_
        self._test_type = test_type
        self._active_change = False
        self._active_warning = False

    def reset(self):
        """Resets the change detector."""
        self._reset_parameters()

    def partial_fit(self, pr_scores: List[bool]):
        """Updates the change detector.

        Args:
            pr_scores: Boolean vector indicating correct predictions.
                If True the prediction by the online learner was correct, False otherwise.
        """
        for pr in pr_scores:
            pr = 1.0 if pr is False else 0.0
            self._active_change = False
            self._active_warning = False

            # 1. UPDATING STATS
            aux_decay_rate = 1 - self._lambda_
            if self._total.EWMA_estimator < 0.0:
                self._total.EWMA_estimator = pr
                self._total.independent_bounded_condition_sum = 1.0
            else:
                self._total.EWMA_estimator = self._lambda_ * pr + aux_decay_rate * self._total.EWMA_estimator
                self._total.independent_bounded_condition_sum = self._lambda_ * self._lambda_ + aux_decay_rate \
                                                                * aux_decay_rate \
                                                                * self._total.independent_bounded_condition_sum

            self._update_incr_statistics(pr=pr)

            if self._monitor_mean_incr(self._drift_confidence):
                self._reset_parameters()
                self._active_change = True
            elif self._monitor_mean_incr(self._warning_confidence):
                self._active_warning = True

            self._update_decr_statistics(pr=pr)

            if self._test_type != 'one-sided' and self._monitor_mean_decr(confidence_level=self._drift_confidence):
                self._reset_parameters()

    def detect_change(self) -> bool:
        """Detects global concept drift."""
        return self._active_change

    def detect_partial_change(self) -> Tuple[bool, list]:
        """Detects partial concept drift.

        Notes:
            HDDMW does not detect partial change.
        """
        return False, []

    def detect_warning_zone(self) -> bool:
        """Detects a warning zone."""
        return self._active_warning

    # ----------------------------------------
    # Tornado Functionality (left unchanged
    # ----------------------------------------
    def _update_incr_statistics(self, pr):
        """Tornado-function (left unchanged)."""
        aux_decay = 1.0 - self._lambda_
        bound = math.sqrt(self._total.independent_bounded_condition_sum * math.log(1.0 / self._drift_confidence, math.e) / 2)

        if self._total.EWMA_estimator + bound < self._incr_cut_point:
            self._incr_cut_point = self._total.EWMA_estimator + bound
            self._sample1_incr_monitoring.EWMA_estimator = self._total.EWMA_estimator
            self._sample1_incr_monitoring.independent_bounded_condition_sum = self._total.independent_bounded_condition_sum
            self._sample2_incr_monitoring = _SampleInfo()
        else:
            if self._sample2_incr_monitoring.EWMA_estimator < 0.0:
                self._sample2_incr_monitoring.EWMA_estimator = pr
                self._sample2_incr_monitoring.independent_bounded_condition_sum = 1.0
            else:
                self._sample2_incr_monitoring.EWMA_estimator = self._lambda_ * pr + aux_decay * self._sample2_incr_monitoring.EWMA_estimator
                self._sample2_incr_monitoring.independent_bounded_condition_sum = self._lambda_ * self._lambda_ + aux_decay * aux_decay * self._sample2_incr_monitoring.independent_bounded_condition_sum

    def _monitor_mean_incr(self, confidence_level):
        """Tornado-function (left unchanged)."""
        return self._detect_mean_increment(self._sample1_incr_monitoring, self._sample2_incr_monitoring, confidence_level)

    @staticmethod
    def _detect_mean_increment(sample_1, sample_2, confidence_level):
        """Tornado-function (left unchanged)."""
        if sample_1.EWMA_estimator < 0.0 or sample_2.EWMA_estimator < 0.0:
            return False
        bound = math.sqrt((sample_1.independent_bounded_condition_sum + sample_2.independent_bounded_condition_sum) * math.log(1 / confidence_level, math.e) / 2)
        return sample_2.EWMA_estimator - sample_1.EWMA_estimator > bound

    def _update_decr_statistics(self, pr):
        """Tornado-function (left unchanged)."""
        aux_decay = 1.0 - self._lambda_
        epsilon = math.sqrt(self._total.independent_bounded_condition_sum * math.log(1.0 / self._drift_confidence, math.e) / 2)

        if self._total.EWMA_estimator - epsilon > self._decr_cut_point:
            self._decr_cut_point = self._total.EWMA_estimator - epsilon
            self._sample1_decr_monitoring.EWMA_estimator = self._total.EWMA_estimator
            self._sample1_decr_monitoring.independent_bounded_condition_sum = self._total.independent_bounded_condition_sum
            self._sample2_decr_monitoring = _SampleInfo()
        else:
            if self._sample2_decr_monitoring.EWMA_estimator < 0.0:
                self._sample2_decr_monitoring.EWMA_estimator = pr
                self._sample2_decr_monitoring.independent_bounded_condition_sum = 1.0
            else:
                self._sample2_decr_monitoring.EWMA_estimator = self._lambda_ * pr + aux_decay * self._sample2_decr_monitoring.EWMA_estimator
                self._sample2_decr_monitoring.independent_bounded_condition_sum = self._lambda_ * self._lambda_ + aux_decay * aux_decay * self._sample2_decr_monitoring.independent_bounded_condition_sum

    def _monitor_mean_decr(self, confidence_level):
        """Tornado-function (left unchanged)."""
        return self._detect_mean_increment(self._sample2_decr_monitoring, self._sample1_decr_monitoring, confidence_level)

    def _reset_parameters(self):
        """Tornado-function (left unchanged)."""
        self._total = _SampleInfo()
        self._sample1_decr_monitoring = _SampleInfo()
        self._sample1_incr_monitoring = _SampleInfo()
        self._sample2_decr_monitoring = _SampleInfo()
        self._sample2_incr_monitoring = _SampleInfo()
        self._incr_cut_point = sys.float_info.max
        self._decr_cut_point = sys.float_info.min


# ----------------------------------------
# Tornado Functionality (left unchanged)
# ----------------------------------------
class _SampleInfo:
    """Tornado-Class (left unchanged)"""
    def __init__(self):
        self.EWMA_estimator = -1.0
        self.independent_bounded_condition_sum = 0.0
