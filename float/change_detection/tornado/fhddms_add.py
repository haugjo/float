from float.change_detection.base_change_detector import BaseChangeDetector
import math


class FHDDMSAdd(BaseChangeDetector):
    """ Additive Stacking Fast Hoeffding Drift Detection Method (FHDDSMAdd)

    Code adopted from https://github.com/alipsgh/tornado, please cite:
    The Tornado Framework
    By Ali Pesaranghader
    University of Ottawa, Ontario, Canada
    E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
    ---
    Paper: Reservoir of Diverse Adaptive Learners and Stacking Fast Hoeffding Drift Detection Methods for Evolving Data Streams
    URL: https://arxiv.org/pdf/1709.02457.pdf

    Attributes:  # Todo: add attribute descriptions
    """
    def __init__(self, evaluation_metrics=None, m=4, n=25, delta=0.000001):
        """ Initialize the concept drift detector

        Args:  # Todo: add argument descriptions
            evaluation_metrics (dict of str: function | dict of str: (function, dict)): {metric_name: metric_function} OR {metric_name: (metric_function, {param_name1: param_val1, ...})} a dictionary of metrics to be used
            m (int):
            n (int):
            delta (float):
        """
        super().__init__(evaluation_metrics, error_based=True)
        self.active_change = False

        self._ELEMENT_SIZE = n
        self._DELTA = delta

        self._stack = []
        self._init_stack(m)

        self._first_round = True
        self._counter = 0
        self._mu_max_short = 0
        self._mu_max_large = 0
        self._num_ones = 0

    def reset(self):
        """ Resets the concept drift detector parameters.
        """
        self._init_stack(len(self._stack))
        self._first_round = True
        self._counter = 0
        self._mu_max_short = 0
        self._mu_max_large = 0
        self._num_ones = 0

    def partial_fit(self, pr):
        """ Update the concept drift detector

        Args:
            pr (bool): indicator of correct prediction (i.e. pr=True) and incorrect prediction (i.e. pr=False)
        """
        self.active_change = False

        self._counter += 1

        if self._counter == (len(self._stack) * self._ELEMENT_SIZE) + 1:
            self._counter -= self._ELEMENT_SIZE
            self._num_ones -= self._stack[0]
            self._stack.pop(0)
            self._stack.append(0.0)
            if self._first_round is True:
                self._first_round = False

        if self._first_round is True:
            index = int(self._counter / self._ELEMENT_SIZE)
            if index == len(self._stack):
                index -= 1
        else:
            index = len(self._stack) - 1

        if pr is True:
            self._stack[index] += 1
            self._num_ones += 1

        # TESTING THE NEW SUB-WINDOWS
        if self._counter % self._ELEMENT_SIZE == 0:
            m_temp = self._stack[index] / self._ELEMENT_SIZE
            if self._mu_max_short < m_temp:
                self._mu_max_short = m_temp
            if self._mu_max_short - m_temp > self.__cal_hoeffding_bound(self._ELEMENT_SIZE):
                self.active_change = True

        # TESTING THE WHOLE WINDOW
        if self._counter == len(self._stack) * self._ELEMENT_SIZE:
            m_temp = self._num_ones / (len(self._stack) * self._ELEMENT_SIZE)
            if self._mu_max_large < m_temp:
                self._mu_max_large = m_temp
            if self._mu_max_large - m_temp > self.__cal_hoeffding_bound(len(self._stack) * self._ELEMENT_SIZE):
                self.active_change = True

    def detected_global_change(self):
        """ Checks whether global concept drift was detected or not.

        Returns:
            bool: whether global concept drift was detected or not.
        """
        return self.active_change

    def detected_warning_zone(self):
        pass

    def detected_partial_change(self):
        pass

    def get_length_estimation(self):
        pass

    # ----------------------------------------
    # Tornado Functionality (left unchanged)
    # ----------------------------------------
    def _init_stack(self, size):
        """
        Tornado-function (left unchanged)
        """
        self._stack.clear()
        for i in range(0, size):
            self._stack.append(0.0)

    def __cal_hoeffding_bound(self, n):
        """
        Tornado-function (left unchanged)
        """
        return math.sqrt(math.log((1 / self._DELTA), math.e) / (2 * n))
