"""Stacking Fast Hoeffding Drift Detection Method.

The source code was adopted from https://github.com/alipsgh/tornado, please cite:
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
---
Original Paper: Reservoir of Diverse Adaptive Learners and Stacking Fast Hoeffding Drift Detection Methods for Evolving Data Streams
URL: https://arxiv.org/pdf/1709.02457.pdf

Copyright (C) 2022 Johannes Haug.
"""
import math
from typing import Tuple, List

from float.change_detection.base_change_detector import BaseChangeDetector


class FHDDMS(BaseChangeDetector):
    """FHDDMS change detector."""
    def __init__(self, m: int = 4, n: int = 25, delta: float = 0.000001, reset_after_drift: bool = False):
        """Inits the change detector.

        Args:
            m: Todo (left unspecified by the Tornado library).
            n: Todo (left unspecified by the Tornado library).
            delta: Todo (left unspecified by the Tornado library).
            reset_after_drift: A boolean indicating if the change detector will be reset after a drift was detected.
        """
        super().__init__(reset_after_drift=reset_after_drift, error_based=True)

        self._WIN = []
        self._WIN_SIZE = m * n
        self._S_WIN_NUM = m
        self._S_WIN_SIZE = n
        self._DELTA = delta
        self._mu_max_short = 0
        self._mu_max_large = 0
        self._active_change = False

    def reset(self):
        """Resets the change detector."""
        self._WIN.clear()
        self._mu_max_short = 0
        self._mu_max_large = 0

    def partial_fit(self, pr_scores: List[bool]):
        """Updates the change detector.

        Args:
            pr_scores:
                A boolean vector indicating correct predictions. 'True' values indicate that the prediction by the
                online learner was correct, otherwise the vector contains 'False'.
        """
        self._active_change = False

        for pr in pr_scores:
            if len(self._WIN) >= self._WIN_SIZE:
                self._WIN.pop(0)
            self._WIN.append(pr)

            if len(self._WIN) == self._WIN_SIZE:
                # TESTING THE SHORT WINDOW
                sub_wins_mu = []
                for i in range(0, self._S_WIN_NUM):
                    sub_win = self._WIN[i * self._S_WIN_SIZE: (i + 1) * self._S_WIN_SIZE]
                    sub_wins_mu.append(sub_win.count(True) / len(sub_win))
                if self._mu_max_short < sub_wins_mu[self._S_WIN_NUM - 1]:
                    self._mu_max_short = sub_wins_mu[self._S_WIN_NUM - 1]
                if self._mu_max_short - sub_wins_mu[self._S_WIN_NUM - 1] > self.__cal_hoeffding_bound(self._S_WIN_SIZE):
                    self._active_change = True

                # TESTING THE LONG WINDOW
                mu_long = sum(sub_wins_mu) / self._S_WIN_NUM
                if self._mu_max_large < mu_long:
                    self._mu_max_large = mu_long
                if self._mu_max_large - mu_long > self.__cal_hoeffding_bound(self._WIN_SIZE):
                    self._active_change = True

    def detect_change(self) -> bool:
        """Detects global concept drift.

        Returns:
            bool: True, if a concept drift was detected, False otherwise.
        """
        return self._active_change

    def detect_partial_change(self) -> Tuple[bool, list]:
        """Detects partial concept drift.

        Notes:
            FHDDMS does not detect partial change.
        """
        return False, []

    def detect_warning_zone(self) -> bool:
        """Detects a warning zone.

        Notes:
            FHDDMS does not raise warnings.
        """
        return False

    # ----------------------------------------
    # Tornado Functionality (left unchanged)
    # ----------------------------------------
    def __cal_hoeffding_bound(self, n):
        """Tornado-function (left unchanged)."""
        return math.sqrt(math.log((1 / self._DELTA), math.e) / (2 * n))
