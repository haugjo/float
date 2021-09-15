from float.concept_drift_detection.concept_drift_detector import ConceptDriftDetector


class PageHinkley(ConceptDriftDetector):
    """ Page Hinkley Drift Detection Method

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
        m_n (int):
        x_mean (float):
        sum (float):
        delta (float):
        lambda_ (int):
        alpha (float):
        prediction_based (bool):
    """
    def __init__(self, evaluation_metrics=None, min_instance=30, delta=0.005, lambda_=50, alpha=1 - 0.0001):
        """ Initialize the Page Hinkley concept drift detector

        Todo: add remaining param descriptions
        Args:
            evaluation_metrics (dict of str: function | dict of str: (function, dict)): {metric_name: metric_function} OR {metric_name: (metric_function, {param_name1: param_val1, ...})} a dictionary of metrics to be used
            min_instance (int):
            delta (float):
            lambda_ (int):
            alpha (float):
        """
        super().__init__(evaluation_metrics)

        self.MIN_NUMBER_INSTANCE = min_instance
        self.m_n = 1
        self.x_mean = 0.0
        self.sum = 0.0
        self.delta = delta
        self.lambda_ = lambda_
        self.alpha = alpha
        self.prediction_based = True  # Todo: this parameter should be part of the super class

    def reset(self):
        """ Resets the concept drift detector parameters.
        """
        self.m_n = 1
        self.x_mean = 0.0
        self.sum = 0.0

    def partial_fit(self, pr):
        """ Update the Page Hinkley test

        Args:
            pr (bool): indicator of correct prediction (i.e. pr=True) and incorrect prediction (i.e. pr=False)
        """
        pr = 1 if pr is False else 0

        self.x_mean = self.x_mean + (pr - self.x_mean) / self.m_n
        self.sum = self.alpha * self.sum + (pr - self.x_mean - self.delta)
        self.m_n += 1

    def detected_global_change(self):
        """ Checks whether global concept drift was detected or not.

        Returns:
            bool: whether global concept drift was detected or not.
        """
        if self.m_n >= self.MIN_NUMBER_INSTANCE:
            if self.sum > self.lambda_:
                return True
            else:
                return False

    def detected_partial_change(self):
        pass

    def detected_warning_zone(self):
        pass

    def get_length_estimation(self):
        pass
