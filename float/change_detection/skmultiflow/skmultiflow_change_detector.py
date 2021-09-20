from float.change_detection.base_change_detector import BaseChangeDetector
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
from skmultiflow.drift_detection import ADWIN, DDM, EDDM, HDDM_A, HDDM_W, KSWIN, PageHinkley


class SkmultiflowChangeDetector(BaseChangeDetector):
    """
    A wrapper for scikit-multiflow drift detection methods.

    Attributes:
        detector (BaseDriftDetector): the concept drift detector
    """
    def __init__(self, detector):
        """
        Initializes the skmultiflow drift detector.

        Args:
            detector (BaseDriftDetector): the concept drift detector
        """
        self.detector = detector
        error_based = self._validate()
        super().__init__(error_based=error_based)

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

    def detected_partial_change(self):
        pass

    def _validate(self):
        """
        Validate the provided skmultiflow drift detector object and return whether it is based on change in error

        :return:
            bool: indicates whether the method requires error measurements from a predictor
        """
        if isinstance(self.detector, (ADWIN, DDM, EDDM, HDDM_A, HDDM_W, PageHinkley)):
            return True
        elif isinstance(self.detector, KSWIN):
            return False
        else:
            raise TypeError("Scikit-Multiflow Drift Detector Class {} is not supported.".format(type(self.detector)))
