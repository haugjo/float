from float.change_detection.base_change_detector import BaseChangeDetector
import sys
import math


class DDM(BaseChangeDetector):
    """ Drift Detection Method (DDM)

    Code adopted from https://github.com/alipsgh/tornado, please cite:
    The Tornado Framework
    By Ali Pesaranghader
    University of Ottawa, Ontario, Canada
    E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
    ---
    Paper: Gama, Joao, et al. "Learning with drift detection."
    Published in: Brazilian Symposium on Artificial Intelligence. Springer, Berlin, Heidelberg, 2004.
    URL: https://link.springer.com/chapter/10.1007/978-3-540-28645-5_29

    Attributes:  # Todo: add attribute descriptions
        min_instance (int):
    """
    def __init__(self, min_instance=30, reset_after_drift=False):
        """ Initialize the concept drift detector

        Todo: add remaining param descriptions
        Args:
            min_instance (int):
            reset_after_drift (bool): indicates whether to reset the change detector after a drift was detected
        """
        super().__init__(reset_after_drift=reset_after_drift, error_based=True)
        self.active_change = False
        self.active_warning = False

        self.MINIMUM_NUM_INSTANCES = min_instance
        self.NUM_INSTANCES_SEEN = 1

        self.__P = 1
        self.__S = 0
        self.__P_min = sys.maxsize
        self.__S_min = sys.maxsize

    def reset(self):
        """ Resets the concept drift detector parameters.
        """
        self.NUM_INSTANCES_SEEN = 1
        self.__P = 1
        self.__S = 0
        self.__P_min = sys.maxsize
        self.__S_min = sys.maxsize

    def partial_fit(self, pr):
        """ Update the concept drift detector

        Args:
            pr (bool): indicator of correct prediction (i.e. pr=True) and incorrect prediction (i.e. pr=False)
        """
        self.active_change = False
        self.active_warning = False

        pr = 1 if pr is False else 0

        # 1. UPDATING STATS
        self.__P += (pr - self.__P) / self.NUM_INSTANCES_SEEN
        self.__S = math.sqrt(self.__P * (1 - self.__P) / self.NUM_INSTANCES_SEEN)

        self.NUM_INSTANCES_SEEN += 1

        if self.NUM_INSTANCES_SEEN < self.MINIMUM_NUM_INSTANCES:
            return

        if self.__P + self.__S <= self.__P_min + self.__S_min:
            self.__P_min = self.__P
            self.__S_min = self.__S

        # 2. UPDATING WARNING AND DRIFT STATUSES
        current_level = self.__P + self.__S
        warning_level = self.__P_min + 2 * self.__S_min
        drift_level = self.__P_min + 3 * self.__S_min

        if current_level > warning_level:
            self.active_warning = True

        if current_level > drift_level:
            self.active_change = True

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
