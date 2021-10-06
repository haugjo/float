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
        comp_times (list): computation time in all time steps
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
        self.comp_times = []

        self.result = dict()
        for measure in measures:  # todo: do we need a _validate_func routine for measures from skmultiflow or river?
            self.result[measure.__name__] = dict()

    def run(self, drifts):
        """
        Updates relevant statistics and computes the evaluation measures in last time step

        Args:
            drifts (list): monitors if there was detected change at each time step
        """
        for measure in self.measures:  # run each evaluation measure
            try:
                if isinstance(self.n_delay, int):  # run single delay parameter
                    mean = measure(self, drifts, self.n_delay)
                    mes = [mean]
                    var = 0
                else:  # run multiple delay parameters
                    mes = []
                    for ndel in self.n_delay:
                        mes.append(measure(self, drifts, ndel))
                    mean = np.mean(mes)
                    var = np.var(mes)

                self.result[measure.__name__]['measures'] = mes
                self.result[measure.__name__]['mean'] = mean
                self.result[measure.__name__]['var'] = var
            except TypeError:
                traceback.print_exc()
                continue