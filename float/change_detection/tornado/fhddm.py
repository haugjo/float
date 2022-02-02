"""Fast Hoeffding Drift Detection Method.

Code adopted from https://github.com/alipsgh/tornado, please cite:
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
---
Paper: Pesaranghader, Ali, and Herna L. Viktor. "Fast hoeffding drift detection method for evolving data streams."
Published in: Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Springer International Publishing, 2016.
URL: https://link.springer.com/chapter/10.1007/978-3-319-46227-1_7

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


class FHDDM(BaseChangeDetector):
    """FHDDM change detector."""
    def __init__(self, n: int = 100, delta: float = 0.000001, reset_after_drift: bool = False):
        """Inits the change detector.

        Args:
            n: Todo
            delta: Todo
            reset_after_drift: See description of base class.
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
            pr_scores: Boolean vector indicating correct predictions.
                If True the prediction by the online learner was correct, False otherwise.
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
        """Detects global concept drift."""
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
