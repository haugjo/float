from float.change_detection.base_change_detector import BaseChangeDetector
import math
import sys


class HDDMW(BaseChangeDetector):
    """ Hoeffding's Bound based Drift Detection Method - W_test Scheme (HDDMW)

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
    def __init__(self, drift_confidence=0.001, warning_confidence=0.005, lambda_=0.05, test_type='one-sided', reset_after_drift=False):
        """ Initialize the concept drift detector

        Args:  # Todo: add argument descriptions
            drift_confidence (float):
            warning_confidence (float):
            lambda_ (float):
            test_type (str):
            reset_after_drift (bool): indicates whether to reset the change detector after a drift was detected
        """
        super().__init__(reset_after_drift=reset_after_drift, error_based=True)
        self.active_change = False
        self.active_warning = False

        self.total = _SampleInfo()
        self.sample1_decr_monitoring = _SampleInfo()
        self.sample1_incr_monitoring = _SampleInfo()
        self.sample2_decr_monitoring = _SampleInfo()
        self.sample2_incr_monitoring = _SampleInfo()
        self.incr_cut_point = sys.float_info.max
        self.decr_cut_point = sys.float_info.min

        self.drift_confidence = drift_confidence
        self.warning_confidence = warning_confidence
        self.lambda_ = lambda_
        self.test_type = test_type

    def reset(self):
        """ Resets the concept drift detector parameters.
        """
        self._reset_parameters()

    def partial_fit(self, pr):
        """ Update the concept drift detector

        Args:
            pr (bool): indicator of correct prediction (i.e. pr=True) and incorrect prediction (i.e. pr=False)
        """
        pr = 1.0 if pr is False else 0.0
        self.active_change = False
        self.active_warning = False

        # 1. UPDATING STATS
        aux_decay_rate = 1 - self.lambda_
        if self.total.EWMA_estimator < 0.0:
            self.total.EWMA_estimator = pr
            self.total.independent_bounded_condition_sum = 1.0
        else:
            self.total.EWMA_estimator = self.lambda_ * pr + aux_decay_rate * self.total.EWMA_estimator
            self.total.independent_bounded_condition_sum = self.lambda_ * self.lambda_ + aux_decay_rate * aux_decay_rate * self.total.independent_bounded_condition_sum

        self._update_incr_statistics(pr)

        if self._monitor_mean_incr(self.drift_confidence):
            self._reset_parameters()
            self.active_change = True
        elif self._monitor_mean_incr(self.warning_confidence):
            self.active_warning = True

        self._update_decr_statistics(pr)

        if self.test_type != 'one-sided' and self._monitor_mean_decr(self.drift_confidence):
            self._reset_parameters()

    def detect_change(self):
        """ Checks whether global concept drift was detected or not.

        Returns:
            bool: whether global concept drift was detected or not.
        """
        return self.active_change

    def detect_warning_zone(self):
        """ Check for Warning Zone

        Returns:
            bool: whether the concept drift detector is in the warning zone or not.
        """
        return self.active_warning

    def detect_partial_change(self):
        return False, None

    def get_length_estimation(self):
        pass

    # ----------------------------------------
    # Tornado Functionality (left unchanged
    # ----------------------------------------
    def _update_incr_statistics(self, pr):
        """
        Tornado-function (left unchanged)
        """
        aux_decay = 1.0 - self.lambda_
        bound = math.sqrt(self.total.independent_bounded_condition_sum * math.log(1.0 / self.drift_confidence, math.e) / 2)

        if self.total.EWMA_estimator + bound < self.incr_cut_point:
            self.incr_cut_point = self.total.EWMA_estimator + bound
            self.sample1_incr_monitoring.EWMA_estimator = self.total.EWMA_estimator
            self.sample1_incr_monitoring.independent_bounded_condition_sum = self.total.independent_bounded_condition_sum
            self.sample2_incr_monitoring = _SampleInfo()
        else:
            if self.sample2_incr_monitoring.EWMA_estimator < 0.0:
                self.sample2_incr_monitoring.EWMA_estimator = pr
                self.sample2_incr_monitoring.independent_bounded_condition_sum = 1.0
            else:
                self.sample2_incr_monitoring.EWMA_estimator = self.lambda_ * pr + aux_decay * self.sample2_incr_monitoring.EWMA_estimator
                self.sample2_incr_monitoring.independent_bounded_condition_sum = self.lambda_ * self.lambda_ + aux_decay * aux_decay * self.sample2_incr_monitoring.independent_bounded_condition_sum

    def _monitor_mean_incr(self, confidence_level):
        """
        Tornado-function (left unchanged)
        """
        return self._detect_mean_increment(self.sample1_incr_monitoring, self.sample2_incr_monitoring, confidence_level)

    @staticmethod
    def _detect_mean_increment(sample_1, sample_2, confidence_level):
        """
        Tornado-function (left unchanged)
        """
        if sample_1.EWMA_estimator < 0.0 or sample_2.EWMA_estimator < 0.0:
            return False
        bound = math.sqrt((sample_1.independent_bounded_condition_sum + sample_2.independent_bounded_condition_sum) * math.log(1 / confidence_level, math.e) / 2)
        return sample_2.EWMA_estimator - sample_1.EWMA_estimator > bound

    def _update_decr_statistics(self, pr):
        """
        Tornado-function (left unchanged)
        """
        aux_decay = 1.0 - self.lambda_
        epsilon = math.sqrt(self.total.independent_bounded_condition_sum * math.log(1.0 / self.drift_confidence, math.e) / 2)

        if self.total.EWMA_estimator - epsilon > self.decr_cut_point:
            self.decr_cut_point = self.total.EWMA_estimator - epsilon
            self.sample1_decr_monitoring.EWMA_estimator = self.total.EWMA_estimator
            self.sample1_decr_monitoring.independent_bounded_condition_sum = self.total.independent_bounded_condition_sum
            self.sample2_decr_monitoring = _SampleInfo()
        else:
            if self.sample2_decr_monitoring.EWMA_estimator < 0.0:
                self.sample2_decr_monitoring.EWMA_estimator = pr
                self.sample2_decr_monitoring.independent_bounded_condition_sum = 1.0
            else:
                self.sample2_decr_monitoring.EWMA_estimator = self.lambda_ * pr + aux_decay * self.sample2_decr_monitoring.EWMA_estimator
                self.sample2_decr_monitoring.independent_bounded_condition_sum = self.lambda_ * self.lambda_ + aux_decay * aux_decay * self.sample2_decr_monitoring.independent_bounded_condition_sum

    def _monitor_mean_decr(self, confidence_level):
        return self._detect_mean_increment(self.sample2_decr_monitoring, self.sample1_decr_monitoring, confidence_level)

    def _reset_parameters(self):
        self.total = _SampleInfo()
        self.sample1_decr_monitoring = _SampleInfo()
        self.sample1_incr_monitoring = _SampleInfo()
        self.sample2_decr_monitoring = _SampleInfo()
        self.sample2_incr_monitoring = _SampleInfo()
        self.incr_cut_point = sys.float_info.max
        self.decr_cut_point = sys.float_info.min


# ----------------------------------------
# Tornado Functionality (left unchanged)
# ----------------------------------------
class _SampleInfo:
    """
    Tornado-Class (left unchanged)
    """
    def __init__(self):
        self.EWMA_estimator = -1.0
        self.independent_bounded_condition_sum = 0.0
