from abc import ABCMeta, abstractmethod
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector


class ConceptDriftDetector(metaclass=ABCMeta):
    """
    Abstract base class which serves as a wrapper for skmultiflow concept drift detection models.

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

    def detected_change(self):
        """
        Checks whether concept drift was detected or not.

        Returns:
            bool: whether the concept drift was detector or not.
        """
        return self.detector.detected_change()

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

    @abstractmethod
    def add_element(self, input_value):
        """
        Adds the relevant data from a sample into the concept drift detector.

        Args:
            input_value (any): whatever input value the concept drift detector takes.
        """
        raise NotImplementedError
