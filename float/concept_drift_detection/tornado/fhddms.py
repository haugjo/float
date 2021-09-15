from float.concept_drift_detection.concept_drift_detector import ConceptDriftDetector
import math


class FHDDMS(ConceptDriftDetector):
    """ Stacking Fast Hoeffding Drift Detection Method (FHDDMS)

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
        super().__init__(evaluation_metrics)
        self.prediction_based = True  # Todo: this parameter should be part of the super class
        self.active_change = False

        self._WIN = []
        self._WIN_SIZE = m * n

        self._S_WIN_NUM = m
        self._S_WIN_SIZE = n
        self._DELTA = delta

        self._mu_max_short = 0
        self._mu_max_large = 0

    def reset(self):
        """ Resets the concept drift detector parameters.
        """
        self._WIN.clear()
        self._mu_max_short = 0
        self._mu_max_large = 0

    def partial_fit(self, pr):
        """ Update the concept drift detector

        Args:
            pr (bool): indicator of correct prediction (i.e. pr=True) and incorrect prediction (i.e. pr=False)
        """
        self.active_change = False

        if len(self._WIN) >= self._WIN_SIZE:
            self._WIN.pop(0)
        self._WIN.append(pr)

        if len(self._WIN) == self._WIN_SIZE:
            # TESTING THE SHORT WINDOW
            sub_wins_mu = []
            for i in range(0, self._S_WIN_NUM):
                sub_win = self._WIN[i * self._S_WIN_SIZE: (i + 1) * self._S_WIN_SIZE]
                sub_wins_mu.append(sub_win.count(True) / len(sub_win))
            if self._mu_max_short < sub_wins_mu[self._S_WIN_NUM - 1]:
                self._mu_max_short = sub_wins_mu[self._S_WIN_NUM - 1]
            if self._mu_max_short - sub_wins_mu[self._S_WIN_NUM - 1] > self.__cal_hoeffding_bound(self._S_WIN_SIZE):
                self.active_change = True

            # TESTING THE LONG WINDOW
            mu_long = sum(sub_wins_mu) / self._S_WIN_NUM
            if self._mu_max_large < mu_long:
                self._mu_max_large = mu_long
            if self._mu_max_large - mu_long > self.__cal_hoeffding_bound(self._WIN_SIZE):
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

    def __cal_hoeffding_bound(self, n):
        """
        Tornado-function (left unchanged)
        """
        return math.sqrt(math.log((1 / self._DELTA), math.e) / (2 * n))
