"""Early Drift Detection Method.

The source code was adopted from https://github.com/alipsgh/tornado, please cite:
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
---
Original Paper: Baena-García, Manuel, et al. "Early drift detection method." (2006).
URL: http://www.cs.upc.edu/~abifet/EDDM.pdf

Copyright (C) 2022 Johannes Haug.
"""
import math
from typing import Tuple, List

from float.change_detection.base_change_detector import BaseChangeDetector


class EDDM(BaseChangeDetector):
    """EDDM change detector."""
    def __init__(self, reset_after_drift: bool = False):
        """Inits the change detector.

        Args:
            reset_after_drift: A boolean indicating if the change detector will be reset after a drift was detected.
        """
        super().__init__(reset_after_drift=reset_after_drift, error_based=True)

        self._WARNING_LEVEL = 0.95
        self._OUT_CONTROL_LEVEL = 0.9
        self._MINIMUM_NUM_INSTANCES = 30
        self._NUM_INSTANCES_SEEN = 0
        self._MINIMUM_NUM_ERRORS = 30
        self._NUM_ERRORS = 0
        self._P = 0.0  # mean
        self._S_TEMP = 0.0
        self._M2S_max = 0
        self._LATEST_E_LOCATION = 0
        self._SECOND_LATEST_E_LOCATION = 0
        self._active_change = False
        self._active_warning = False

    def reset(self):
        """Resets the change detector."""
        self._P = 0.0
        self._S_TEMP = 0.0
        self._NUM_ERRORS = 0
        self._M2S_max = 0
        self._LATEST_E_LOCATION = 0
        self._SECOND_LATEST_E_LOCATION = 0
        self._NUM_INSTANCES_SEEN = 0

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
            self._NUM_INSTANCES_SEEN += 1

            if pr is False:
                self._NUM_ERRORS += 1

                self._SECOND_LATEST_E_LOCATION = self._LATEST_E_LOCATION
                self._LATEST_E_LOCATION = self._NUM_INSTANCES_SEEN
                distance = self._LATEST_E_LOCATION - self._SECOND_LATEST_E_LOCATION

                old_p = self._P
                self._P += (distance - self._P) / self._NUM_ERRORS
                self._S_TEMP += (distance - self._P) * (distance - old_p)

                s = math.sqrt(self._S_TEMP / self._NUM_ERRORS)
                m2s = self._P + 2 * s

                if self._NUM_INSTANCES_SEEN > self._MINIMUM_NUM_INSTANCES:
                    if m2s > self._M2S_max:
                        self._M2S_max = m2s
                    elif self._NUM_ERRORS > self._MINIMUM_NUM_ERRORS:
                        r = m2s / self._M2S_max
                        if r < self._WARNING_LEVEL:
                            self._active_warning = True
                        if r < self._OUT_CONTROL_LEVEL:
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
            EDDM does not detect partial change.
        """
        return False, []

    def detect_warning_zone(self) -> bool:
        """Detects a warning zone.

        Returns:
            bool: True, if the change detector has detected a warning zone, False otherwise.
        """
        return self._active_warning
