"""Fast Hoeffding Drift Detection Method.

The source code was adopted from https://github.com/alipsgh/tornado, please cite:
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
---
Original Paper: Pesaranghader, Ali, and Herna L. Viktor. "Fast hoeffding drift detection method for evolving data streams."
Published in: Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Springer International Publishing, 2016.
URL: https://link.springer.com/chapter/10.1007/978-3-319-46227-1_7

Copyright (C) 2022 Johannes Haug.
"""
import math
from typing import Tuple, List

from float.change_detection.base_change_detector import BaseChangeDetector


class FHDDM(BaseChangeDetector):
    """FHDDM change detector."""
    def __init__(self, n: int = 100, delta: float = 0.000001, reset_after_drift: bool = False):
        """Inits the change detector.

        Args:
            n: Todo (left unspecified by the Tornado library).
            delta: Todo (left unspecified by the Tornado library).
            reset_after_drift: A boolean indicating if the change detector will be reset after a drift was detected.
        """
        super().__init__(reset_after_drift=reset_after_drift, error_based=True)

        self.__DELTA = delta
        self.__N = n
        self.__E = math.sqrt(math.log((1 / self.__DELTA), math.e) / (2 * self.__N))
        self.__WIN = []
        self.__MU_M = 0
        self._active_change = False

    def reset(self):
        """Resets the change detector."""
        self.__WIN.clear()
        self.__MU_M = 0

    def partial_fit(self, pr_scores: List[bool]):
        """Updates the change detector.

        Args:
            pr_scores:
                A boolean vector indicating correct predictions. 'True' values indicate that the prediction by the
                online learner was correct, otherwise the vector contains 'False'.
        """
        self._active_change = False

        for pr in pr_scores:
            if len(self.__WIN) >= self.__N:
                self.__WIN.pop(0)
            self.__WIN.append(pr)

            if len(self.__WIN) >= self.__N:
                mu_t = self.__WIN.count(True) / self.__N
                if self.__MU_M < mu_t:
                    self.__MU_M = mu_t
                self._active_change = (self.__MU_M - mu_t) > self.__E

    def detect_change(self) -> bool:
        """Detects global concept drift.

        Returns:
            bool: True, if a concept drift was detected, False otherwise.
        """
        return self._active_change

    def detect_partial_change(self) -> Tuple[bool, list]:
        """Detects partial concept drift.

        Notes:
            FHDDM does not detect partial change.
        """
        return False, []

    def detect_warning_zone(self) -> bool:
        """Detects a warning zone.

        Notes:
            FHDDM does not raise warnings.
        """
        return False
