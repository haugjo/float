from abc import ABCMeta, abstractmethod
import numpy as np


class ConceptDriftDetector(metaclass=ABCMeta):
    """
    Abstract base class for concept drift detection models.

    Attributes:
        global_drifts (list): monitors if there was detected change at each time step
        comp_times (list): computation time in all time steps
    """
    def __init__(self):
        """
        Initializes the concept drift detector.
        """
        self.global_drifts = []
        self.comp_times = []
        self.average_delay = None

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

    def evaluate(self, time_step, max_n_samples, known_drifts, batch_size):
        """
        Evaluates the concept drift detector at one time step.

        Args:
            time_step (int): the current time step
            max_n_samples (int): the maximum number of samples used for the evaluation
            known_drifts (list): the known drifts for the data stream
            batch_size (int, int): the batch size used for the data stream
        """
        if self.detected_global_change():
            if time_step not in self.global_drifts:
                self.global_drifts.append(time_step)

        self.average_delay = self.get_average_delay(max_n_samples, known_drifts, batch_size)

    def get_average_delay(self, max_n_samples, known_drifts, batch_size):
        """
        Returns the average delay between a known drift and the detection of the concept drift detector.

        Args:
            max_n_samples (int): the maximum number of samples used for the evaluation
            known_drifts (list): the known drifts for the data stream
            batch_size (int, int): the batch size used for the data stream

        Returns:
            float: the average delay between known drift and detected drift
        """
        detected_drifts = np.asarray(self.global_drifts) * batch_size + batch_size

        total_delay = 0
        for known_drift in known_drifts:
            if isinstance(known_drift, tuple):
                drift_start = known_drift[0]
            else:
                drift_start = known_drift

            if len(detected_drifts[detected_drifts >= drift_start]) > 0:
                total_delay += detected_drifts[detected_drifts >= drift_start][0] - drift_start
            else:
                total_delay += max_n_samples - drift_start

        return total_delay / (batch_size * len(known_drifts))
