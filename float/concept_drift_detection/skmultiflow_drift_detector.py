from float.concept_drift_detection.concept_drift_detector import ConceptDriftDetector
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
from skmultiflow.drift_detection.kswin import KSWIN


class SkmultiflowDriftDetector(ConceptDriftDetector):
    """
    Serves as a wrapper for the skmultiflow drift_detection module.

    Attributes:
        detector (BaseDriftDetector): the concept drift detector
    """
    def __init__(self, detector, evaluation_metrics=None):
        """
        Initializes the skmultiflow drift detector.

        Args:
            detector (BaseDriftDetector): the concept drift detector
            evaluation_metrics (dict of str: function | dict of str: (function, dict)): {metric_name: metric_function} OR {metric_name: (metric_function, {param_name1: param_val1, ...})} a dictionary of metrics to be used
        """
        super().__init__(evaluation_metrics)
        self.detector = detector
        self.prediction_based = False if type(self.detector) is KSWIN else True

    def reset(self):
        """
        Resets the concept drift detector parameters.
        """
        self.detector.reset()

    def detected_global_change(self):
        """
        Checks whether global concept drift was detected or not.

        Returns:
            bool: whether global concept drift was detected or not.
        """
        return self.detector.detected_change()

    def detected_partial_change(self):
        """
        Checks whether partial concept drift was detected or not.

        Returns:
            bool: whether partial concept drift was detected or not.
        """
        return False

    def detected_warning_zone(self):
        """
        If the concept drift detector supports the warning zone, this function will return
        whether it's inside the warning zone or not.

        Returns:
            bool: whether the concept drift detector is in the warning zone or not.
        """
        return self.detector.in_warning_zone

    def get_length_estimation(self):
        """
        Returns the length estimation.

        Returns:
            float: length estimation
        """
        return self.detector.estimation

    def partial_fit(self, input_value, *args, **kwargs):
        """
        Update the parameters of the concept drift detection model.

        Args:
            input_value (any): whatever input value the concept drift detector takes.
        """
        self.detector.add_element(input_value)
