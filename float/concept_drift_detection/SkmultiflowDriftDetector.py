from float.concept_drift_detection.ConceptDriftDetector import ConceptDriftDetector
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector


class SkmultiflowDriftDetector(ConceptDriftDetector):
    """
    Serves as a wrapper for the skmultiflow drift_detection module.

    Attributes:
        detector (BaseDriftDetector): the concept drift detector
    """
    def __init__(self, detector):
        """
        Receives a skmultiflow BaseDriftDetector object.

        Args:
            detector (BaseDriftDetector): the concept drift detector
        """
        self.detector = detector

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
        self.detector.detected_change()

    def detected_partial_change(self):
        """
        Checks whether partial concept drift was detected or not.

        Returns:
            bool: whether partial concept drift was detected or not.
        """
        pass

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
            int: length estimation
        """
        return self.detector.estimation

    def partial_fit(self, input_value):
        """
        Update the parameters of the concept drift detection model.

        Args:
            input_value (any): whatever input value the concept drift detector takes.
        """
        self.detector.add_element(input_value)
