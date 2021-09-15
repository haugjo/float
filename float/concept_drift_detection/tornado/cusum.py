from float.concept_drift_detection.concept_drift_detector import ConceptDriftDetector


class Cusum(ConceptDriftDetector):
    """ Cumulative Sum (Cusum) Drift Detection Method

    Code adopted from https://github.com/alipsgh/tornado, please cite:
    The Tornado Framework
    By Ali Pesaranghader
    University of Ottawa, Ontario, Canada
    E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
    ---
    Paper: Page, Ewan S. "Continuous inspection schemes."
    Published in: Biometrika 41.1/2 (1954): 100-115.
    URL: http://www.jstor.org/stable/2333009

    Attributes:  # Todo: add attribute descriptions
        min_instance (int):
        delta (float):
        lambda_ (int):
    """
    def __init__(self, evaluation_metrics=None, min_instance=30, delta=0.005, lambda_=50):
        """ Initialize the concept drift detector

        Todo: add remaining param descriptions
        Args:
            evaluation_metrics (dict of str: function | dict of str: (function, dict)): {metric_name: metric_function} OR {metric_name: (metric_function, {param_name1: param_val1, ...})} a dictionary of metrics to be used
            min_instance (int):
            delta (float):
            lambda_ (int):
        """
        super().__init__(evaluation_metrics)
        self.prediction_based = True  # Todo: this parameter should be part of the super class
        self.active_change = False

        self.MINIMUM_NUM_INSTANCES = min_instance

        self.m_n = 1
        self.x_mean = 0
        self.sum = 0
        self.delta = delta
        self.lambda_ = lambda_

    def reset(self):
        """ Resets the concept drift detector parameters.
        """
        self.m_n = 1
        self.x_mean = 0
        self.sum = 0

    def partial_fit(self, pr):
        """ Update the concept drift detector

        Args:
            pr (bool): indicator of correct prediction (i.e. pr=True) and incorrect prediction (i.e. pr=False)
        """
        pr = 1 if pr is False else 0

        self.active_change = False

        # 1. UPDATING STATS
        self.x_mean = self.x_mean + (pr - self.x_mean) / self.m_n
        self.sum = max([0, self.sum + pr - self.x_mean - self.delta])
        self.m_n += 1

        # 2. UPDATING WARNING AND DRIFT STATUSES
        if self.m_n >= self.MINIMUM_NUM_INSTANCES:
            if self.sum > self.lambda_:
                self.active_change = True

    def detected_global_change(self):
        """ Checks whether global concept drift was detected or not.

        Returns:
            bool: whether global concept drift was detected or not.
        """
        return self.active_change

    def detected_partial_change(self):
        pass

    def detected_warning_zone(self):
        pass

    def get_length_estimation(self):
        pass
