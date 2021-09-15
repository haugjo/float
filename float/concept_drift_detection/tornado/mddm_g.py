from float.concept_drift_detection.concept_drift_detector import ConceptDriftDetector
import math


class MDDMG(ConceptDriftDetector):
    """ McDiarmid Drift Detection Method - Geometric Scheme (MDDMG)

    Code adopted from https://github.com/alipsgh/tornado, please cite:
    The Tornado Framework
    By Ali Pesaranghader
    University of Ottawa, Ontario, Canada
    E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
    ---
    Paper: Pesaranghader, Ali, et al. "McDiarmid Drift Detection Method for Evolving Data Streams."
    Published in: International Joint Conference on Neural Network (IJCNN 2018)
    URL: https://arxiv.org/abs/1710.02030

    Attributes:  # Todo: add attribute descriptions
        min_instance (int):
    """
    def __init__(self, evaluation_metrics=None, n=100, ratio=1.01, delta=0.000001):
        """ Initialize the concept drift detector

        Todo: add remaining param descriptions
        Args:
            evaluation_metrics (dict of str: function | dict of str: (function, dict)): {metric_name: metric_function} OR {metric_name: (metric_function, {param_name1: param_val1, ...})} a dictionary of metrics to be used
            n (int):
            ratio (float):
            delta (float):
        """
        super().__init__(evaluation_metrics)
        self.prediction_based = True  # Todo: this parameter should be part of the super class
        self.active_change = False

        self.win = []
        self.n = n
        self.ratio = ratio
        self.delta = delta

        self.e = math.sqrt(0.5 * self._cal_sigma() * (math.log(1 / self.delta, math.e)))
        self.u_max = 0

    def reset(self):
        """ Resets the concept drift detector parameters.
        """
        self.win.clear()
        self.u_max = 0

    def partial_fit(self, pr):
        """ Update the concept drift detector

        Args:
            pr (bool): indicator of correct prediction (i.e. pr=True) and incorrect prediction (i.e. pr=False)
        """
        self.active_change = False

        if len(self.win) == self.n:
            self.win.pop(0)
        self.win.append(pr)

        if len(self.win) == self.n:
            u = self._cal_w_sigma()
            self.u_max = u if u > self.u_max else self.u_max
            self.active_change = True if (self.u_max - u > self.e) else False

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

    def _cal_sigma(self):
        """
        Tornado-function (left unchanged)
        """
        sum_, bound_sum, r = 0, 0, self.ratio
        for i in range(self.n):
            sum_ += r
            r *= self.ratio
        r = self.ratio
        for i in range(self.n):
            bound_sum += math.pow(r / sum_, 2)
            r *= self.ratio
        return bound_sum

    def _cal_w_sigma(self):
        """
        Tornado-function (left unchanged)
        """
        total_sum, win_sum, r = 0, 0, self.ratio
        for i in range(self.n):
            total_sum += r
            win_sum += self.win[i] * r
            r *= self.ratio
        return win_sum / total_sum