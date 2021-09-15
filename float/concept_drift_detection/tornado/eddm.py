from float.concept_drift_detection.concept_drift_detector import ConceptDriftDetector
import math


class EDDM(ConceptDriftDetector):
    """ Early Drift Detection Method (DDM)

    Code adopted from https://github.com/alipsgh/tornado, please cite:
    The Tornado Framework
    By Ali Pesaranghader
    University of Ottawa, Ontario, Canada
    E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
    ---
    Paper: Baena-García, Manuel, et al. "Early drift detection method." (2006).
    URL: http://www.cs.upc.edu/~abifet/EDDM.pdf

    Attributes:  # Todo: add attribute descriptions
    """
    def __init__(self, evaluation_metrics=None):
        """ Initialize the concept drift detector

        Args:
            evaluation_metrics (dict of str: function | dict of str: (function, dict)): {metric_name: metric_function} OR {metric_name: (metric_function, {param_name1: param_val1, ...})} a dictionary of metrics to be used
        """
        super().__init__(evaluation_metrics)
        self.prediction_based = True  # Todo: this parameter should be part of the super class
        self.active_change = False
        self.active_warning = False

        self.WARNING_LEVEL = 0.95
        self.OUT_CONTROL_LEVEL = 0.9

        self.MINIMUM_NUM_INSTANCES = 30
        self.NUM_INSTANCES_SEEN = 0

        self.MINIMUM_NUM_ERRORS = 30
        self.NUM_ERRORS = 0

        self.P = 0.0  # mean
        self.S_TEMP = 0.0
        self.M2S_max = 0

        self.LATEST_E_LOCATION = 0
        self.SECOND_LATEST_E_LOCATION = 0

    def reset(self):
        """ Resets the concept drift detector parameters.
        """
        self.P = 0.0
        self.S_TEMP = 0.0
        self.NUM_ERRORS = 0
        self.M2S_max = 0

        self.LATEST_E_LOCATION = 0
        self.SECOND_LATEST_E_LOCATION = 0

        self.NUM_INSTANCES_SEEN = 0

    def partial_fit(self, prediction_status):
        """ Update the concept drift detector

        Args:
            prediction_status (bool): indicator of correct prediction (i.e. pr=True) and incorrect prediction (i.e. pr=False)
        """
        self.active_change = False
        self.active_warning = False

        self.NUM_INSTANCES_SEEN += 1

        if prediction_status is False:

            self.NUM_ERRORS += 1

            self.SECOND_LATEST_E_LOCATION = self.LATEST_E_LOCATION
            self.LATEST_E_LOCATION = self.NUM_INSTANCES_SEEN
            distance = self.LATEST_E_LOCATION - self.SECOND_LATEST_E_LOCATION

            old_p = self.P
            self.P += (distance - self.P) / self.NUM_ERRORS
            self.S_TEMP += (distance - self.P) * (distance - old_p)

            s = math.sqrt(self.S_TEMP / self.NUM_ERRORS)
            m2s = self.P + 2 * s

            if self.NUM_INSTANCES_SEEN > self.MINIMUM_NUM_INSTANCES:
                if m2s > self.M2S_max:
                    self.M2S_max = m2s
                elif self.NUM_ERRORS > self.MINIMUM_NUM_ERRORS:
                    r = m2s / self.M2S_max
                    if r < self.WARNING_LEVEL:
                        self.active_warning = True
                    if r < self.OUT_CONTROL_LEVEL:
                        self.active_change = True

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
        pass

    def get_length_estimation(self):
        pass
