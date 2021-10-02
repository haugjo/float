from float.change_detection.base_change_detector import BaseChangeDetector
import math


class EWMA(BaseChangeDetector):
    """ Exponentially Weigthed Moving Average (EWMA) Drift Detection Method

    Code adopted from https://github.com/alipsgh/tornado, please cite:
    The Tornado Framework
    By Ali Pesaranghader
    University of Ottawa, Ontario, Canada
    E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
    ---
    Paper: Ross, Gordon J., et al. "Exponentially weighted moving average charts for detecting concept drift."
    Published in: Pattern Recognition Letters 33.2 (2012): 191-198.
    URL: https://arxiv.org/pdf/1212.6018.pdf

    Attributes:  # Todo: add attribute descriptions
    """
    def __init__(self, min_instance=30, lambda_=0.2, reset_after_drift=False):
        """ Initialize the concept drift detector

        Args:
            min_instance (int):
            lambda_ (float):
            reset_after_drift (bool): indicates whether to reset the change detector after a drift was detected
        """
        super().__init__(reset_after_drift=reset_after_drift, error_based=True)
        self.active_change = False
        self.active_warning = False

        self.MINIMUM_NUM_INSTANCES = min_instance

        self.m_n = 1.0
        self.m_sum = 0.0
        self.m_p = 0.0
        self.m_s = 0.0
        self.z_t = 0.0
        self.lambda_ = lambda_

    def reset(self):
        """ Resets the concept drift detector parameters.
        """
        self.m_n = 1
        self.m_sum = 0
        self.m_p = 0
        self.m_s = 0
        self.z_t = 0

    def partial_fit(self, pr):
        """ Update the concept drift detector

        Args:
            pr (bool): indicator of correct prediction (i.e. pr=True) and incorrect prediction (i.e. pr=False)
        """
        pr = 1 if pr is False else 0

        self.active_change = False
        self.active_warning = False

        # 1. UPDATING STATS
        self.m_sum += pr
        self.m_p = self.m_sum / self.m_n
        self.m_s = math.sqrt(
            self.m_p * (1.0 - self.m_p) * self.lambda_ * (1.0 - math.pow(1.0 - self.lambda_, 2.0 * self.m_n)) / (
                        2.0 - self.lambda_))
        self.m_n += 1

        self.z_t += self.lambda_ * (pr - self.z_t)
        L_t = 3.97 - 6.56 * self.m_p + 48.73 * math.pow(self.m_p, 3) - 330.13 * math.pow(self.m_p, 5) \
              + 848.18 * math.pow(self.m_p, 7)

        # 2. UPDATING WARNING AND DRIFT STATUSES
        if self.m_n < self.MINIMUM_NUM_INSTANCES:
            return

        if self.z_t > self.m_p + L_t * self.m_s:
            self.active_change = True
        elif self.z_t > self.m_p + 0.5 * L_t * self.m_s:
            self.active_warning = True

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
