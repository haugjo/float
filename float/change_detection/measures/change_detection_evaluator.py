from abc import ABCMeta
import traceback
import numpy as np


class ChangeDetectionEvaluator(metaclass=ABCMeta):
    """
    Abstract base class for change detection evaluation measures

    Attributes:
        measures (list): list of evaluation measure functions
        known_drifts (list): positions in dataset corresponding to known concept drift
        batch_size (int): no. of observations processed per iteration/time step
        n_samples (int): total number of observations
        n_delay (int | list): no. of observations after a known concept drift in which to count detections as true positive
        n_init_tolerance (int): no. of observations used for initial training, not counted for measure
        result (dict): dictionary of results per evaluation measure
    """
    def __init__(self, measures, known_drifts, batch_size, n_samples, n_delay=100, n_init_tolerance=100):
        """ Initialize change detection evaluation measure

        Args:
            measures (list): list of evaluation measure functions
            known_drifts (list): positions in dataset corresponding to known concept drift
            batch_size (int): no of observations processed per iteration/time step
            n_samples (int): total number of observations
            n_delay (int | list): no. of observations after a known concept drift in which to count detections as true positive
            n_init_tolerance (int): no. of observations used for initial training, not counted for measure
        """
        self.measures = measures
        self.known_drifts = known_drifts
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.n_delay = n_delay
        self.n_init_tolerance = n_init_tolerance

        self.result = dict()
        for measure in measures:
            self.result[measure.__name__] = dict()

    def run(self, global_drifts):
        """
        Updates relevant statistics and computes the evaluation measures in last time step

        Args:
            global_drifts (list): monitors if there was detected change at each time step
        """
        for measure in self.measures:  # run each evaluation measure
            try:
                if isinstance(self.n_delay, int):  # run single delay parameter
                    mean = measure(self, global_drifts, self.n_delay)
                    std = 0
                else:  # run multiple delay parameters
                    res = []
                    for ndel in self.n_delay:
                        res.append(measure(self, global_drifts, ndel))
                    mean = np.mean(res)
                    std = np.std(res)

                self.result[measure.__name__]['mean'] = mean
                self.result[measure.__name__]['std'] = std
            except TypeError:
                traceback.print_exc()
                continue
