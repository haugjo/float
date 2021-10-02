from float.change_detection.base_change_detector import BaseChangeDetector
import math
import sys


class RDDM(BaseChangeDetector):
    """ Reactive Drift Detection Method (RDDM)

    Code adopted from https://github.com/alipsgh/tornado, please cite:
    The Tornado Framework
    By Ali Pesaranghader
    University of Ottawa, Ontario, Canada
    E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
    ---
    Paper: Barros, Roberto, et al. "RDDM: Reactive drift detection method."
    Published in: Expert Systems with Applications. Elsevier, 2017.
    URL: https://www.sciencedirect.com/science/article/pii/S0957417417305614

    Attributes:  # Todo: add attribute descriptions
        min_instance (int):
    """
    def __init__(self, min_instance=129, warning_level=1.773, drift_level=2.258,
                 max_size_concept=40000, min_size_stable_concept=7000, warn_limit=1400, reset_after_drift=False):
        """ Initialize the concept drift detector

        Todo: add remaining param descriptions
        Args:
            min_instance (int):
            warning_level (float):
            drift_level (float):
            max_size_concept (int):
            min_size_stable_concept (int):
            warn_limit (int):
            reset_after_drift (bool): indicates whether to reset the change detector after a drift was detected
        """
        super().__init__(reset_after_drift=reset_after_drift, error_based=True)
        self.active_change = False
        self.active_warning = False

        self.min_num_instance = min_instance
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.max_concept_size = max_size_concept
        self.min_size_stable_concept = min_size_stable_concept
        self.warn_limit = warn_limit

        self.m_n = 1
        self.m_p = 1
        self.m_s = 0
        self.m_p_min = sys.maxsize
        self.m_s_min = sys.maxsize
        self.m_p_s_min = sys.maxsize

        self.stored_predictions = [0 for _ in range(self.min_size_stable_concept)]
        self.num_stored_instances = 0
        self.first_pos = 0
        self.last_pos = -1
        self.last_warn_pos = -1
        self.last_warn_inst = -1
        self.inst_num = 0
        self.rddm_drift = False
        self.is_change_detected = False
        self.is_warning_zone = False

    def reset(self):
        """ Resets the concept drift detector parameters.
        """
        self.m_n = 1
        self.m_p = 1
        self.m_s = 0
        self.m_p_min = sys.maxsize
        self.m_s_min = sys.maxsize
        self.m_p_s_min = sys.maxsize

    def partial_fit(self, pr):
        """ Update the concept drift detector

        Args:
            pr (bool): indicator of correct prediction (i.e. pr=True) and incorrect prediction (i.e. pr=False)
        """
        pr = 1 if pr is False else 0

        self.active_change = False
        self.active_warning = False

        if self.rddm_drift:  #
            self._reset_rddm()  #
            if self.last_warn_pos != -1:  #
                self.first_pos = self.last_warn_pos  #
                self.num_stored_instances = self.last_pos - self.first_pos + 1  #
                if self.num_stored_instances <= 0:  #
                    self.num_stored_instances += self.min_size_stable_concept  #

            pos = self.first_pos  #
            for i in range(0, self.num_stored_instances):  #
                self.m_p += ((self.stored_predictions[pos] - self.m_p) / self.m_n)  #
                self.m_s = math.sqrt(self.m_p * (1 - self.m_p) / self.m_n)
                if self.is_change_detected and (self.m_n > self.min_num_instance) and (
                        self.m_p + self.m_s < self.m_p_s_min):
                    self.m_p_min = self.m_p
                    self.m_s_min = self.m_s
                    self.m_p_s_min = self.m_p + self.m_s
                self.m_n += 1
                pos = (pos + 1) % self.min_size_stable_concept

            self.last_warn_pos = -1
            self.last_warn_inst = -1
            self.rddm_drift = False
            self.is_change_detected = False

        self.last_pos = (self.last_pos + 1) % self.min_size_stable_concept
        self.stored_predictions[self.last_pos] = pr
        if self.num_stored_instances < self.min_size_stable_concept:
            self.num_stored_instances += 1
        else:
            self.first_pos = (self.first_pos + 1) % self.min_size_stable_concept
            if self.last_warn_pos == self.last_pos:
                self.last_warn_pos = -1

        self.m_p += (pr - self.m_p) / self.m_n
        self.m_s = math.sqrt(self.m_p * (1 - self.m_p) / self.m_n)

        self.inst_num += 1
        self.m_n += 1
        self.is_warning_zone = False

        if self.m_n <= self.min_num_instance:
            return

        if self.m_p + self.m_s < self.m_p_s_min:
            self.m_p_min = self.m_p
            self.m_s_min = self.m_s
            self.m_p_s_min = self.m_p + self.m_s

        if self.m_p + self.m_s > self.m_p_min + self.drift_level * self.m_s_min:
            self.is_change_detected, self.active_change = True, True
            self.rddm_drift = True
            if self.last_warn_inst == -1:
                self.first_pos = self.last_pos
                self.num_stored_instances = 1
            return

        if self.m_p + self.m_s > self.m_p_min + self.warning_level * self.m_s_min:
            if (self.last_warn_inst != -1) and (self.last_warn_inst + self.warn_limit <= self.inst_num):
                self.is_change_detected, self.active_change = True, True
                self.rddm_drift = True
                self.first_pos = self.last_pos
                self.num_stored_instances = 1
                self.last_warn_pos = -1
                self.last_warn_inst = -1
                return

            self.is_warning_zone, self.active_warning = True, True
            if self.last_warn_inst == -1:
                self.last_warn_inst = self.inst_num
                self.last_warn_pos = self.last_pos
        else:
            self.last_warn_inst = -1
            self.last_warn_pos = -1

        if self.m_n > self.max_concept_size and self.is_warning_zone is False:
            self.rddm_drift = True

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
    def _reset_rddm(self):
        """
        Tornado-function (left unchanged)
        """
        self.m_n = 1
        self.m_p = 1
        self.m_s = 0
        if self.is_change_detected:
            self.m_p_min = sys.maxsize
            self.m_s_min = sys.maxsize
            self.m_p_s_min = sys.maxsize
