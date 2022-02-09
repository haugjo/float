"""Drift Detection Method.

The source code was adopted from https://github.com/alipsgh/tornado, please cite:
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
---
Original Paper: Gama, Joao, et al. "Learning with drift detection."
Published in: Brazilian Symposium on Artificial Intelligence. Springer, Berlin, Heidelberg, 2004.
URL: https://link.springer.com/chapter/10.1007/978-3-540-28645-5_29

Copyright (C) 2022 Johannes Haug.
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
            min_instance: Todo (left unspecified by the Tornado library).
            reset_after_drift: A boolean indicating if the change detector will be reset after a drift was detected.
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
            pr_scores:
                A boolean vector indicating correct predictions. 'True' values indicate that the prediction by the
                online learner was correct, otherwise the vector contains 'False'.
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
        """Detects global concept drift.

        Returns:
            bool: True, if a concept drift was detected, False otherwise.
        """
        return self._active_change

    def detect_partial_change(self) -> Tuple[bool, list]:
        """Detects partial concept drift.

        Notes:
            DDM does not detect partial change.
        """
        return False, []

    def detect_warning_zone(self) -> bool:
        """Detects a warning zone.

        Returns:
            bool: True, if the change detector has detected a warning zone, False otherwise.
        """
        return self._active_warning
