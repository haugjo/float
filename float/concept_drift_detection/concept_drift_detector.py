from abc import ABCMeta, abstractmethod
import numpy as np


class ConceptDriftDetector(metaclass=ABCMeta):
    """
    Abstract base class for concept drift detection models.

    Attributes:
        global_drifts (list): monitors if there was detected change at each time step
        comp_times (list): computation time in all time steps
        average_delay (float): average delay with which a known concept drift is detected
        true_positive_rates (list): true positive rate for different delay ranges
        false_discovery_rates (list): false discovery rate for different delay ranges
        precision_scores (list): precision for different delay ranges
    """
    def __init__(self, max_delay_range):
        """
        Initializes the concept drift detector.

        Args:
            max_delay_range (int): maximum delay for which TPR, FDR and precision should be computed
        """
        self.global_drifts = []
        self.comp_times = []
        self.average_delay = None
        self.true_positive_rates = []
        self.false_discovery_rates = []
        self.precision_scores = []
        self.max_delay_range = max_delay_range

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

    def evaluate(self, time_step, max_n_samples, known_drifts, batch_size, last_iteration):
        """
        Evaluates the concept drift detector at one time step.

        Args:
            time_step (int): the current time step
            max_n_samples (int): the maximum number of samples used for the evaluation
            known_drifts (list): the known drifts for the data stream
            batch_size (int): the batch size used for the data stream
            last_iteration (bool): True if this is the last iteration of the pipeline, False otherwise
        """
        if self.detected_global_change():
            if time_step not in self.global_drifts:
                self.global_drifts.append(time_step)

        if known_drifts and last_iteration:
            self.average_delay = self.get_average_delay(max_n_samples, known_drifts, batch_size)
            self.true_positive_rates, self.false_discovery_rates, self.precision_scores = self.get_tpr_fdr_and_precision(
                known_drifts, batch_size)

    def get_average_delay(self, max_n_samples, known_drifts, batch_size):
        """
        Returns the average delay between a known drift and the detection of the concept drift detector.

        Args:
            max_n_samples (int): the maximum number of samples used for the evaluation
            known_drifts (list): the known drifts for the data stream
            batch_size (int): the batch size used for the data stream

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

    def get_tpr_fdr_and_precision(self, known_drifts, batch_size):
        """
        Computes the true positive rate, false discovery rate and precision for different delay ranges.

        Args:
            known_drifts (list): the known drifts for the data stream
            batch_size (int): the batch size used for the data stream

        Returns:
            (list, list, list): the TPR, FDR and precision for different delay ranges respectively
        """
        tpr = []
        fdr = []
        precision = []
        training_tolerance = 80
        delay_range = np.arange(1, self.max_delay_range + 1, 1)
        detected_drifts = np.asarray(self.global_drifts) * batch_size + batch_size
        relevant_drifts = detected_drifts[detected_drifts > training_tolerance * batch_size]

        for delay in delay_range:
            true_positives = 0
            false_positives = 0
            for known_drift in known_drifts:
                if isinstance(known_drift, tuple):
                    drift_start = known_drift[0]
                    drift_end = known_drift[1]
                else:
                    drift_start = known_drift
                    drift_end = known_drift

                tp_bool = np.logical_and(relevant_drifts >= drift_start,
                                         relevant_drifts <= drift_end + delay * batch_size)
                if tp_bool.any():
                    true_positives += 1

            false_positives += len(relevant_drifts) - true_positives

            if len(relevant_drifts) > 0:
                tpr.append((delay, true_positives / len(known_drifts)))
                fdr.append((delay, false_positives / len(relevant_drifts)))
                precision_i = 1 - fdr[-1][1]
                precision.append((delay, precision_i))
            else:
                tpr.append((delay, 0))
                fdr.append((delay, None))
                precision.append((delay, None))

        return tpr, fdr, precision
