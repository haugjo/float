"""Change Detection Evaluator.

This module contains an evaluator class for active change (i.e. concept drift) detection methods.

Copyright (C) 2022 Johannes Haug.
"""
import numpy as np
import traceback
from typing import Callable, List, Union


class ChangeDetectionEvaluator:
    """Change detection evaluation class.

    This class is required to compute the performance measures and store the corresponding results in the evaluation
    of the change detection method.

    Attributes:
        measure_funcs (List[Callable]): A list of evaluation measure functions.
        known_drifts (List[int] | List[tuple]):
            The positions in the dataset (indices) corresponding to known concept drifts.
        batch_size (int): The number of observations processed per iteration/time step.
        n_total (int): The total number of observations.
        n_delay (int | list):
                The number of observations after a known concept drift, during which we count the detections made by
                the model as true positives. If the argument is a list, the evaluator computes results for each delay
                specified in the list.
        n_init_tolerance (int):
            The number of observations reserved for the initial training. We do not consider these observations in the
            evaluation.
        comp_times (list): Computation times for updating the change detector per time step.
        memory_changes (list):
            Memory changes (in GB RAM) per training iteration of the change detector.
        result (dict): Results (i.e. calculated measurements, mean, and variance) for each evaluation measure function.
    """
    def __init__(self,
                 measure_funcs: List[Callable],
                 known_drifts: Union[List[int], List[tuple]],
                 batch_size: int,
                 n_total: int,
                 n_delay: Union[int, list] = 100,
                 n_init_tolerance: int = 100):
        """Initializes the change detection evaluation object.

        Args:
            measure_funcs: A list of evaluation measure functions.
            known_drifts:
                The positions in the dataset (indices) corresponding to known concept drifts.
            batch_size: The number of observations processed per iteration/time step.
            n_total: The total number of observations.
            n_delay:
                The number of observations after a known concept drift, during which we count the detections made by
                the model as true positives. If the argument is a list, the evaluator computes results for each delay
                specified in the list.
            n_init_tolerance:
                The number of observations reserved for the initial training. We do not consider these observations in
                the evaluation.
        """
        self.measure_funcs = measure_funcs
        self.known_drifts = known_drifts
        self.batch_size = batch_size
        self.n_total = n_total
        self.n_delay = n_delay
        self.n_init_tolerance = n_init_tolerance
        self.comp_times = []
        self.memory_changes = []

        self.result = dict()
        for measure_func in measure_funcs:
            self.result[measure_func.__name__] = dict()

    def run(self, drifts: List):
        """Computes the evaluation measures.

        Other than the PredictionEvaluator and FeatureSelectionEvaluator, the ChangeDetectionEvaluator is only run
        once at the end of the evaluation.

        Args:
            drifts: List of time steps corresponding to detected concept drifts.

        Raises:
            TypeError: Error while executing the provided evaluation measure functions.
        """
        for measure_func in self.measure_funcs:
            try:
                if isinstance(self.n_delay, int):  # Run with a single delay parameter
                    mean = measure_func(evaluator=self, drifts=drifts, n_delay=self.n_delay)
                    mes = [mean]
                    var = 0
                else:  # Run with multiple delay parameters
                    mes = []
                    for ndel in self.n_delay:
                        mes.append(measure_func(evaluator=self, drifts=drifts, n_delay=ndel))
                    mean = np.nanmean(mes)
                    var = np.nanvar(mes)

                self.result[measure_func.__name__]['measures'] = mes
                self.result[measure_func.__name__]['mean'] = mean
                self.result[measure_func.__name__]['var'] = var
            except TypeError:
                traceback.print_exc()
                continue
