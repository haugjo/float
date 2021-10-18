"""Evaluation Module for Online Feature Selection Methods.

This module contains an evaluator class for online feature selection methods.

Copyright (C) 2021 Johannes Haug

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np
from numpy.typing import ArrayLike
import traceback
from typing import Optional, List, Callable


class FeatureSelectionEvaluator:
    """Online feature selection evaluator class.

    Attributes:
        measure_funcs (List[Callable]): List of evaluation measure functions.
        decay_rate (float |None):
            If this parameter is set, the measurements are additionally aggregated with a decay/fading factor.
        window_size (int | None):
            If this parameter is set, the measurements are additionally aggregated in a sliding window.
        comp_times (list): List of computation times per iteration of feature weighting and selection.
        memory_changes (list):
            List of measured memory changes (GB RAM) per training iteration of the online feature selection model.
        result (dict): The raw and aggregated measurements of each evaluation measure function.
    """
    def __init__(self, measure_funcs: List[Callable], decay_rate: Optional[float] = None,
                 window_size: Optional[int] = None):
        """Inits the online feature selection evaluation object.

        Args:
            measure_funcs: List of evaluation measure functions.
            decay_rate:
                If this parameter is set, the measurements are additionally aggregated with a decay/fading factor.
            window_size: If this parameter is set, the measurements are additionally aggregated in a sliding window.
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

    def run(self, selected_features: ArrayLike, n_total_features: int):
        """Updates relevant statistics and computes the evaluation measures.

        Args:
            selected_features (ArrayLike): The indices of all currently selected features.
            n_total_features (int): The total number of features.

        Raises:
            TypeError: If the calculation of a measure runs an error.
        """
        for measure_func in self.measure_funcs:
            try:
                new_measure_val = measure_func(selected_features, n_total_features)
                self.result[measure_func.__name__]['measures'].append(new_measure_val)
                self.result[measure_func.__name__]['mean'].append(np.mean(self.result[measure_func.__name__]['measures']))
                self.result[measure_func.__name__]['var'].append(np.var(self.result[measure_func.__name__]['measures']))

                if self.decay_rate:
                    if len(self.result[measure_func.__name__]['mean_decay']) > 0:
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
                        np.mean(self.result[measure_func.__name__]['measures'][-self.window_size:])
                    )
                    self.result[measure_func.__name__]['var_window'].append(
                        np.var(self.result[measure_func.__name__]['measures'][-self.window_size:])
                    )
            except TypeError:
                traceback.print_exc()
                continue
