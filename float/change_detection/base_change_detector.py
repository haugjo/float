from abc import ABCMeta, abstractmethod


class BaseChangeDetector(metaclass=ABCMeta):
    """
    Abstract base class for change detection models.

    Attributes:
        error_based (bool): indicates whether change detector relies on error measures obtained from a predictor
        global_drifts (list): time steps where a global concept drift was detected
        partial_drifts (list): time steps and features where a partial concept drift was detected
        warnings (list): time steps where a global warning was issued
    """
    def __init__(self, reset_after_drift, error_based=False):
        """
        Initializes the concept drift detector.

        Args:
            reset_after_drift (bool): indicates whether to reset the change detector after a drift was detected
            error_based (bool): indicates whether change detector relies on error measures obtained from a predictor
        """
        self.reset_after_drift = reset_after_drift
        self.error_based = error_based

        self.global_drifts = []
        self.partial_drifts = []
        self.warnings = []

    @abstractmethod
    def reset(self):
        """
        Resets the concept drift detector.
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
            list: indices of input features with detected drift
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
