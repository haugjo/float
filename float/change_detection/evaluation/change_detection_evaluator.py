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
import numpy as np
import traceback
from typing import Callable, List, Union


class ChangeDetectionEvaluator:
    """Change detection evaluation class.

    This class computes and stores the measure/metric functions and results for the evaluation of explicit
    change detection methods.

    Attributes:
        measure_funcs (List[Callable]): A list of evaluation measure functions.
        known_drifts (List[int] | List[tuple]):
            The positions in the dataset (indices) corresponding to known concept drifts.
        batch_size (int): The number of observations processed per iteration/time step.
        n_total (int): The total number of observations.
        n_pretrain (int | None): Number of observations used for the initial training of the predictive model.
        n_delay (int | list): The number of observations after a known concept drift, during which we count
                the detections made by the model as true positives. If the argument is a list, the evaluator computes
                results for each delay specified in the list.
        n_init_tolerance (int): The number of observations reserved for the initial training. We do not consider
            these observations in the evaluation.
        comp_times (list): Computation times of updating the change detector per time step.
        memory_changes (list):
            List of measured memory changes (GB RAM) per training iteration of the concept drift detector.
        result (dict): Results (i.e. calculated measurements, mean, and variance) for each evaluation measure function
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
            measure_funcs (List[Callable]): A list of evaluation measure functions.
            known_drifts (List[int] | List[tuple]):
                The positions in the dataset (indices) corresponding to known concept drifts.
            batch_size (int): The number of observations processed per iteration/time step.
            n_total (int): The total number of observations.
            n_delay (int | list): The number of observations after a known concept drift, during which we count
                the detections made by the model as true positives. If the argument is a list, the evaluator computes
                results for each delay specified in the list.
            n_init_tolerance (int): The number of observations reserved for the initial training. We do not consider
                these observations in the evaluation.
        """
        self.measure_funcs = measure_funcs
        self.known_drifts = known_drifts
        self.batch_size = batch_size
        self.n_total = n_total
        self.n_pretrain = None
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
            TypeError: If there occurs an error while executing the provided evaluation measure function.
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

    def correct_known_drifts(self):
        """Corrects the known drift positions if we do pre-training.

        We Subtract 'n_pretrain' from all known drift positions, as these observations are not considered in the actual
        pipeline run.
        """
        if self.n_pretrain is not None and self.n_pretrain > 0:
            corrected_drifts = []
            for drift in self.known_drifts:
                if isinstance(drift, tuple):
                    corrected_drifts.append((drift[0] - self.n_pretrain, drift[1] - self.n_pretrain))
                else:
                    corrected_drifts.append(drift - self.n_pretrain)
            self.known_drifts = corrected_drifts
