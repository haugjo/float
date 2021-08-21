from abc import ABCMeta, abstractmethod
import numpy as np
import traceback


class ConceptDriftDetector(metaclass=ABCMeta):
    """
    Abstract base class for concept drift detection models.

    Attributes:
        evaluation (dict of str: list[float]): a dictionary of metric names and their corresponding metric values as lists
        global_drifts (list): monitors if there was detected change at each time step
        comp_times (list): computation time in all time steps
        average_delay (float): average delay with which a known concept drift is detected
        true_positive_rates (list): true positive rate for different delay ranges
        false_discovery_rates (list): false discovery rate for different delay ranges
        precision_scores (list): precision for different delay ranges
    """
    def __init__(self, evaluation_metrics):
        """
        Initializes the concept drift detector.

        Args:
            evaluation_metrics (dict of str: function | dict of str: (function, dict)): {metric_name: metric_function} OR {metric_name: (metric_function, {param_name1: param_val1, ...})} a dictionary of metrics to be used
        """
        self.evaluation_metrics = evaluation_metrics
        self.evaluation = {key: [] for key in self.evaluation_metrics.keys()}

        self.global_drifts = []
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

    def evaluate(self, time_step, last_iteration):
        """
        Evaluates the concept drift detector at one time step.

        Args:
            time_step (int): the current time step
            last_iteration (bool): True if this is the last iteration of the pipeline, False otherwise
        """
        if self.detected_global_change():
            if time_step not in self.global_drifts:
                self.global_drifts.append(time_step)

        # TODO generalize to other types of metrics
        if last_iteration:
            for metric_name in self.evaluation:
                if isinstance(self.evaluation_metrics[metric_name], tuple):
                    metric_func = self.evaluation_metrics[metric_name][0]
                    metric_params = self.evaluation_metrics[metric_name][1]
                else:
                    metric_func = self.evaluation_metrics[metric_name]
                    metric_params = {}
                try:
                    metric_val = metric_func(self.global_drifts, **metric_params)
                except TypeError:
                    if time_step == 0:
                        traceback.print_exc()
                    continue

                self.evaluation[metric_name] = metric_val

    @staticmethod
    def get_average_delay(global_drifts, known_drifts, batch_size, max_n_samples):
        """
        Returns the average delay between a known drift and the detection of the concept drift detector.

        Args:
            global_drifts (list): the detected global drifts
            known_drifts (list): the known drifts for the data stream
            batch_size (int): the batch size used for the data stream
            max_n_samples (int): the maximum number of samples used for the evaluation

        Returns:
            float: the average delay between known drift and detected drift
        """
        detected_drifts = np.asarray(global_drifts) * batch_size + batch_size

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

    # TODO make this more efficient
    @staticmethod
    def get_tpr_fdr_and_precision(global_drifts, known_drifts, batch_size, max_delay_range):
        """
        Computes the true positive rate, false discovery rate and precision for different delay ranges.

        Args:
            global_drifts (list): the detected global drifts
            known_drifts (list): the known drifts for the data stream
            batch_size (int): the batch size used for the data stream
            max_delay_range (int): maximum delay for which TPR, FDR and precision should be computed

        Returns:
            (list, list, list): the TPR, FDR and precision for different delay ranges respectively
        """
        tpr = []
        fdr = []
        precision = []
        training_tolerance = 80
        delay_range = np.arange(1, max_delay_range + 1, 1)
        detected_drifts = np.asarray(global_drifts) * batch_size + batch_size
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
                tpr.append(true_positives / len(known_drifts))
                fdr.append(false_positives / len(relevant_drifts))
                precision_i = 1 - fdr[-1]
                precision.append(precision_i)
            else:
                tpr.append(0)
                fdr.append(None)
                precision.append(None)

        return tpr, fdr, precision

    @staticmethod
    def get_tpr(global_drifts, known_drifts, batch_size, max_delay_range):
        tpr, _, _ = ConceptDriftDetector.get_tpr_fdr_and_precision(global_drifts, known_drifts, batch_size, max_delay_range)
        return tpr

    @staticmethod
    def get_fdr(global_drifts, known_drifts, batch_size, max_delay_range):
        _, fdr, _ = ConceptDriftDetector.get_tpr_fdr_and_precision(global_drifts, known_drifts, batch_size, max_delay_range)
        return fdr

    @staticmethod
    def get_precision(global_drifts, known_drifts, batch_size, max_delay_range):
        _, _, precision = ConceptDriftDetector.get_tpr_fdr_and_precision(global_drifts, known_drifts, batch_size, max_delay_range)
        return precision
