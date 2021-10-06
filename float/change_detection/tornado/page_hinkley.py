from float.change_detection.base_change_detector import BaseChangeDetector


class PageHinkley(BaseChangeDetector):
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
    def __init__(self, min_instance=30, delta=0.005, lambda_=50, alpha=1 - 0.0001, reset_after_drift=False):
        """ Initialize the concept drift detector

        Todo: add remaining param descriptions
        Args:
            min_instance (int):
            delta (float):
            lambda_ (int):
            alpha (float):
            reset_after_drift (bool): indicates whether to reset the change detector after a drift was detected
        """
        super().__init__(reset_after_drift=reset_after_drift, error_based=True)
        self.active_change = False

        self.MINIMUM_NUM_INSTANCES = min_instance
        self.m_n = 1
        self.x_mean = 0.0
        self.sum = 0.0
        self.delta = delta
        self.lambda_ = lambda_
        self.alpha = alpha

    def reset(self):
        """ Resets the concept drift detector parameters.
        """
        self.m_n = 1
        self.x_mean = 0.0
        self.sum = 0.0

    def partial_fit(self, pr):
        """ Update the concept drift detector

        Args:
            pr (bool): indicator of correct prediction (i.e. pr=True) and incorrect prediction (i.e. pr=False)
        """
        pr = 1 if pr is False else 0

        self.active_change = False

        # 1. UPDATING STATS
        self.x_mean = self.x_mean + (pr - self.x_mean) / self.m_n
        self.sum = self.alpha * self.sum + (pr - self.x_mean - self.delta)
        self.m_n += 1

        # 2. UPDATING WARNING AND DRIFT STATUSES
        if self.m_n >= self.MINIMUM_NUM_INSTANCES:
            if self.sum > self.lambda_:
                self.active_change = True

    def detect_change(self):
        """ Checks whether global concept drift was detected or not.

        Returns:
            bool: whether global concept drift was detected or not.
        """
        return self.active_change

    def detect_partial_change(self):
        return False, None

    def detect_warning_zone(self):
        return False
