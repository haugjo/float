from abc import ABCMeta, abstractmethod


class ConceptDriftDetector(metaclass=ABCMeta):
    """
    Abstract base class for concept drift detection models.

    Attributes:
        change_detections (list): monitors if there was detected change at each time step
        comp_times (list): computation time in all time steps
    """
    def __init__(self):
        """
        Initializes the concept drift detector.
        """
        self.change_detections = []
        self.comp_times = []

    @abstractmethod
    def reset(self):
        """
        Resets the concept drift detector parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def detected_global_change(self):
        """
        Checks whether global concept drift was detected or not.

        Returns:
            bool: whether global concept drift was detected or not.
        """
        raise NotImplementedError

    @abstractmethod
    def detected_partial_change(self):
        """
        Checks whether partial concept drift was detected or not.

        Returns:
            bool: whether partial concept drift was detected or not.
        """
        raise NotImplementedError

    @abstractmethod
    def detected_warning_zone(self):
        """
        If the concept drift detector supports the warning zone, this function will return
        whether it's inside the warning zone or not.

        Returns:
            bool: whether the concept drift detector is in the warning zone or not.
        """
        raise NotImplementedError

    @abstractmethod
    def get_length_estimation(self):
        """
        Returns the length estimation.

        Returns:
            int: length estimation
        """
        raise NotImplementedError

    @abstractmethod
    def partial_fit(self, *args, **kwargs):
        """
        Update the parameters of the concept drift detection model.
        """
        raise NotImplementedError

    def evaluate(self):
        """
        Evaluates the concept drift detector at one time step.
        """
        self.change_detections.append(self.detected_global_change())

