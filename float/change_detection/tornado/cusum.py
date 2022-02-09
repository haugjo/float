"""Cumulative Sum Drift Detection Method.

The source code was adopted from https://github.com/alipsgh/tornado, please cite:
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
---
Original Paper: Page, Ewan S. "Continuous inspection schemes."
Published in: Biometrika 41.1/2 (1954): 100-115.
URL: http://www.jstor.org/stable/2333009

Copyright (C) 2022 Johannes Haug.
"""
from typing import Tuple, List

from float.change_detection.base_change_detector import BaseChangeDetector


class Cusum(BaseChangeDetector):
    """Cusum Change Detector."""
    def __init__(self, min_instance: int = 30, delta: float = 0.005, lambda_: int = 50,
                 reset_after_drift: bool = False):
        """Inits the change detector.

        Args:
            min_instance: Todo (left unspecified by the Tornado library).
            delta: Todo (left unspecified by the Tornado library).
            lambda_: Todo (left unspecified by the Tornado library).
            reset_after_drift: A boolean indicating if the change detector will be reset after a drift was detected.
        """
        super().__init__(reset_after_drift=reset_after_drift, error_based=True)

        self._MINIMUM_NUM_INSTANCES = min_instance
        self._m_n = 1
        self._x_mean = 0
        self._sum = 0
        self._delta = delta
        self._lambda_ = lambda_
        self._active_change = False

    def reset(self):
        """Resets the change detector."""
        self._m_n = 1
        self._x_mean = 0
        self._sum = 0

    def partial_fit(self, pr_scores: List[bool]):
        """Updates the change detector.

        Args:
            pr_scores:
                A boolean vector indicating correct predictions. 'True' values indicate that the prediction by the
                online learner was correct, otherwise the vector contains 'False'.
        """
        for pr in pr_scores:
            pr = 1 if pr is False else 0

            self._active_change = False

            # 1. UPDATING STATS
            self._x_mean = self._x_mean + (pr - self._x_mean) / self._m_n
            self._sum = max([0, self._sum + pr - self._x_mean - self._delta])
            self._m_n += 1

            # 2. UPDATING WARNING AND DRIFT STATUSES
            if self._m_n >= self._MINIMUM_NUM_INSTANCES:
                if self._sum > self._lambda_:
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
            Cusum does not detect partial change.
        """
        return False, []

    def detect_warning_zone(self) -> bool:
        """Detects a warning zone.

        Notes:
            Cusum does not raise warnings.
        """
        return False
