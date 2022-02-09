"""Online Feature Selection Evaluator.

This module contains an evaluator class for online feature selection methods.

Copyright (C) 2022 Johannes Haug.
"""
import numpy as np
import traceback
from typing import Optional, List, Callable


class FeatureSelectionEvaluator:
    """Online feature selection evaluator class.

    Attributes:
        measure_funcs (List[Callable]): List of evaluation measure functions.
        decay_rate (float |None):
            If this parameter is not None, the measurements are additionally aggregated with the specific decay/fading
            factor.
        window_size (int | None):
            If this parameter is not None, the measurements are additionally aggregated in a sliding window.
        comp_times (list): List of computation times per iteration of feature weighting and selection.
        memory_changes (list):
            Memory changes (in GB RAM) per training iteration of the online feature selection model.
        result (dict): The raw and aggregated measurements of each evaluation measure function.
    """
    def __init__(self,
                 measure_funcs: List[Callable],
                 decay_rate: Optional[float] = None,
                 window_size: Optional[int] = None):
        """Inits the online feature selection evaluation object.

        Args:
            measure_funcs: List of evaluation measure functions.
            decay_rate:
                If this parameter is not None, the measurements are additionally aggregated with the specific
                decay/fading factor.
            window_size:
                If this parameter is not None, the measurements are additionally aggregated in a sliding window.
        """
        self.measure_funcs = measure_funcs
        self.decay_rate = decay_rate
        self.window_size = window_size
        self.comp_times = []
        self.memory_changes = []

        self.result = dict()
        for measure_func in measure_funcs:
            self.result[measure_func.__name__] = dict()
            self.result[measure_func.__name__]['measures'] = []
            self.result[measure_func.__name__]['mean'] = []
            self.result[measure_func.__name__]['var'] = []

            if self.decay_rate:
                self.result[measure_func.__name__]['mean_decay'] = []
                self.result[measure_func.__name__]['var_decay'] = []

            if self.window_size:
                self.result[measure_func.__name__]['mean_window'] = []
                self.result[measure_func.__name__]['var_window'] = []

    def run(self, selected_features_history: List[list], n_total_features: int):
        """Updates relevant statistics and computes the evaluation measures.

        Args:
            selected_features_history: A list of all selected feature vectors obtained over time.
            n_total_features: The total number of features.

        Raises:
            TypeError: If the calculation of a measure runs an error.
        """
        for measure_func in self.measure_funcs:
            try:
                new_measure_val = measure_func(selected_features_history, n_total_features)
                self.result[measure_func.__name__]['measures'].append(new_measure_val)
                self.result[measure_func.__name__]['mean'].append(np.nanmean(self.result[measure_func.__name__]['measures']))
                self.result[measure_func.__name__]['var'].append(np.nanvar(self.result[measure_func.__name__]['measures']))

                if self.decay_rate:
                    if len(self.result[measure_func.__name__]['mean_decay']) > 0 \
                            and not np.isnan(self.result[measure_func.__name__]['mean_decay'][-1]):
                        delta = new_measure_val - self.result[measure_func.__name__]['mean_decay'][-1]
                        self.result[measure_func.__name__]['mean_decay'].append(
                            self.result[measure_func.__name__]['mean_decay'][-1] + self.decay_rate * delta
                        )
                        self.result[measure_func.__name__]['var_decay'].append(
                            (1 - self.decay_rate) * (self.result[measure_func.__name__]['var_decay'][-1] + self.decay_rate * delta ** 2)
                        )
                    else:
                        self.result[measure_func.__name__]['mean_decay'].append(new_measure_val)
                        self.result[measure_func.__name__]['var_decay'].append(.0)

                if self.window_size:
                    self.result[measure_func.__name__]['mean_window'].append(
                        np.nanmean(self.result[measure_func.__name__]['measures'][-self.window_size:])
                    )
                    self.result[measure_func.__name__]['var_window'].append(
                        np.nanvar(self.result[measure_func.__name__]['measures'][-self.window_size:])
                    )
            except TypeError:
                traceback.print_exc()
                continue
