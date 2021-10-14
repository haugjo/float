"""Evaluation Module for Change Detection Methods.

This module contains an evaluator class for explicit change (i.e. concept drift) detection methods.

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
from abc import ABCMeta
import numpy as np
import traceback
from typing import Callable, List, Union, Dict


class ChangeDetectionEvaluator(metaclass=ABCMeta):
    """Change detection evaluation class.

    This class computes and stores the measure/metric functions and results for the evaluation of explicit
    change detection methods.

    Attributes:
        measure_funcs (List[Callable]): A list of evaluation measure functions.
        known_drifts (List[int] | List[tuple]):
            The positions in the dataset (indices) corresponding to known concept drifts.
        batch_size (int): The number of observations processed per iteration/time step.
        n_samples (int): Te total number of observations.
        n_delay (int | list): The number of observations after a known concept drift, during which we count
                the detections made by the model as true positives. If the argument is a list, the evaluator computes
                results for each delay specified in the list.
        n_init_tolerance (int): The number of observations reserved for the initial training. We do not consider
            these observations in the evaluation.
        comp_times (list): Computation times of the change detector per time step.
        result (dict): Results (i.e. calculated measurements, mean, and variance) for each evaluation measure function
    """
    def __init__(self, measure_funcs: List[Callable], known_drifts: Union[List[int], List[tuple]], batch_size: int,
                 n_samples: int, n_delay: Union[int, list] = 100, n_init_tolerance: int = 100):
        """ Initializes the change detection evaluation measure.

        Args:
            measure_funcs (List[Callable]): A list of evaluation measure functions.
            known_drifts (List[int] | List[tuple]):
                The positions in the dataset (indices) corresponding to known concept drifts.
            batch_size (int): The number of observations processed per iteration/time step.
            n_samples (int): The total number of observations.
            n_delay (int | list): The number of observations after a known concept drift, during which we count
                the detections made by the model as true positives. If the argument is a list, the evaluator computes
                results for each delay specified in the list.
            n_init_tolerance (int): The number of observations reserved for the initial training. We do not consider
                these observations in the evaluation.
        """
        self.measure_funcs = measure_funcs
        self.known_drifts = known_drifts
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.n_delay = n_delay
        self.n_init_tolerance = n_init_tolerance
        self.comp_times = []

        self.result = dict()
        for measure_func in measure_funcs:  # Todo: To support scikit-multiflow/river functionality, add a validation function
            self.result[measure_func.__name__] = dict()

    def run(self, drifts: List):
        """Updates relevant statistics and computes the evaluation measures.

        Args:
            drifts: List of time steps corresponding to detected concept drifts.

        Raises:
            TypeError: If there occurs an error while executing the provided evaluation measure function.
        """
        for measure_func in self.measure_funcs:
            try:
                if isinstance(self.n_delay, int):  # run with a single delay parameter
                    mean = measure_func(evaluator=self, drifts=drifts, n_delay=self.n_delay)
                    mes = [mean]
                    var = 0
                else:  # run with multiple delay parameters
                    mes = []
                    for ndel in self.n_delay:
                        mes.append(measure_func(evaluator=self, drifts=drifts, n_delay=ndel))
                    mean = np.mean(mes)
                    var = np.var(mes)

                self.result[measure_func.__name__]['measures'] = mes
                self.result[measure_func.__name__]['mean'] = mean
                self.result[measure_func.__name__]['var'] = var
            except TypeError:
                traceback.print_exc()
                continue
