from float.change_detection.base_change_detector import BaseChangeDetector
import math


class HDDMA(BaseChangeDetector):
    """ Hoeffding's Bound based Drift Detection Method - A_test Scheme (HDDMA)

    Code adopted from https://github.com/alipsgh/tornado, please cite:
    The Tornado Framework
    By Ali Pesaranghader
    University of Ottawa, Ontario, Canada
    E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
    ---
    Paper: Frías-Blanco, Isvani, et al. "Online and non-parametric drift detection methods based on Hoeffding’s bounds."
    Published in: IEEE Transactions on Knowledge and Data Engineering 27.3 (2015): 810-823.
    URL: http://ieeexplore.ieee.org/abstract/document/6871418/

    Attributes:  # Todo: add attribute descriptions
    """
    def __init__(self, drift_confidence=0.001, warning_confidence=0.005, test_type='two-sided', reset_after_drift=False):
        """ Initialize the concept drift detector

        Args:  # Todo: add argument descriptions
            drift_confidence (float):
            warning_confidence (float):
            test_type (str):
            reset_after_drift (bool): indicates whether to reset the change detector after a drift was detected
        """
        super().__init__(reset_after_drift=reset_after_drift, error_based=True)
        self.active_change = False
        self.active_warning = False

        self.drift_confidence = drift_confidence
        self.warning_confidence = warning_confidence
        self.test_type = test_type

        self.n_min = 0
        self.c_min = 0
        self.total_n = 0
        self.total_c = 0
        self.n_max = 0
        self.c_max = 0

    def reset(self):
        """ Resets the concept drift detector parameters.
        """
        self.n_min = 0
        self.c_min = 0
        self.total_n = 0
        self.total_c = 0
        self.n_max = 0
        self.c_max = 0

    def partial_fit(self, pr):
        """ Update the concept drift detector

        Args:
            pr (bool): indicator of correct prediction (i.e. pr=True) and incorrect prediction (i.e. pr=False)
        """
        pr = 1 if pr is False else 0

        self.active_change = False
        self.active_warning = False

        # 1. UPDATING STATS
        self.total_n += 1
        self.total_c += pr

        if self.n_min == 0:
            self.n_min = self.total_n
            self.c_min = self.total_c

        if self.n_max == 0:
            self.n_max = self.total_n
            self.c_max = self.total_c

        cota = math.sqrt((1.0 / (2 * self.n_min)) * math.log(1.0 / self.drift_confidence, math.e))
        cota1 = math.sqrt((1.0 / (2 * self.total_n)) * math.log(1.0 / self.drift_confidence, math.e))
        if self.c_min / self.n_min + cota >= self.total_c / self.total_n + cota1:
            self.c_min = self.total_c
            self.n_min = self.total_n

        cota = math.sqrt((1.0 / (2 * self.n_max)) * math.log(1.0 / self.drift_confidence, math.e))
        if self.c_max / self.n_max - cota <= self.total_c / self.total_n - cota1:
            self.c_max = self.total_c
            self.n_max = self.total_n

        if self._mean_incr(self.drift_confidence):
            self.n_min = self.n_max = self.total_n = 0
            self.c_min = self.c_max = self.total_c = 0
            self.active_change = True
        elif self._mean_incr(self.warning_confidence):
            self.active_warning = True

        # 2. UPDATING WARNING AND DRIFT STATUSES
        if self.test_type == 'two-sided' and self._mean_decr():
            self.n_min = self.n_max = self.total_n = 0
            self.c_min = self.c_max = self.total_c = 0

    def detected_global_change(self):
        """ Checks whether global concept drift was detected or not.

        Returns:
            bool: whether global concept drift was detected or not.
        """
        return self.active_change

    def detected_warning_zone(self):
        """ Check for Warning Zone

        Returns:
            bool: whether the concept drift detector is in the warning zone or not.
        """
        return self.active_warning

    def detected_partial_change(self):
        return False, None

    def get_length_estimation(self):
        pass

    # ----------------------------------------
    # Tornado Functionality (left unchanged)
    # ----------------------------------------
    def _mean_incr(self, confidence_level):
        """
        Tornado-function (left unchanged)
        """
        if self.n_min == self.total_n:
            return False
        m = (self.total_n - self.n_min) / self.n_min * (1.0 / self.total_n)
        cota = math.sqrt((m / 2) * math.log(2.0 / confidence_level, math.e))
        return self.total_c / self.total_n - self.c_min / self.n_min >= cota

    def _mean_decr(self):
        """
        Tornado-function (left unchanged)
        """
        if self.n_max == self.total_n:
            return False
        m = (self.total_n - self.n_max) / self.n_max * (1.0 / self.total_n)
        cota = math.sqrt((m / 2) * math.log(2.0 / self.drift_confidence, math.e))
        return self.c_max / self.n_max - self.total_c / self.total_n >= cota
