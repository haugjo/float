"""Predictive Model Evaluator.

This module contains an evaluator class for online predictive models.

Copyright (C) 2022 Johannes Haug.
"""
import numpy as np
from numpy.typing import ArrayLike
from numpy.random import Generator
import inspect
import traceback
from typing import List, Callable, Optional

from float.prediction import BasePredictor


class PredictionEvaluator:
    """Online prediction evaluator class.

    Attributes:
        measure_funcs (List[Callable]): List of evaluation measure functions.
        decay_rate (float |None):
            If this parameter is not None, the measurements are additionally aggregated with the specific decay/fading
            factor.
        window_size (int | None):
            If this parameter is not None, the measurements are additionally aggregated in a sliding window.
        kwargs (dict):
            A dictionary containing additional and specific keyword arguments, which are passed to the evaluation
            functions.
        testing_comp_times (list): List of computation times per testing iteration.
        training_comp_times (list): List of computation times per training iteration.
        memory_changes (list):
            Memory changes (in GB RAM) per training iteration of the online feature selection model.
        result (dict):
            The raw and aggregated measurements of each evaluation measure function.
    """

    def __init__(self,
                 measure_funcs: List[Callable],
                 decay_rate: Optional[float] = None,
                 window_size: Optional[float] = None,
                 **kwargs):
        """Inits the prediction evaluation object.

        Args:
            measure_funcs: List of evaluation measure functions.
            decay_rate:
                If this parameter is not None, the measurements are additionally aggregated with the specific
                decay/fading factor.
            window_size:
                If this parameter is not None, the measurements are additionally aggregated in a sliding window.
            kwargs:
                A dictionary containing additional and specific keyword arguments, which are passed to the evaluation
                functions.
        """
        self.measure_funcs = measure_funcs
        self.decay_rate = decay_rate
        self.window_size = window_size
        self.kwargs = kwargs

        self.testing_comp_times = []
        self.training_comp_times = []
        self.memory_changes = []

        self.result = dict()
        for measure_func in measure_funcs:
            self._validate_func(measure_func, kwargs)

            measure_name = type(self.kwargs['metric']).__name__.lower() if measure_func.__name__ == 'river_classification_metric' else measure_func.__name__
            self.result[measure_name] = dict()
            self.result[measure_name]['measures'] = []
            self.result[measure_name]['mean'] = []
            self.result[measure_name]['var'] = []

            if self.decay_rate:
                self.result[measure_name]['mean_decay'] = []
                self.result[measure_name]['var_decay'] = []

            if self.window_size:
                self.result[measure_name]['mean_window'] = []
                self.result[measure_name]['var_window'] = []

    def run(self, y_true: ArrayLike, y_pred: ArrayLike, X: ArrayLike, predictor: BasePredictor, rng: Generator):
        """Updates relevant statistics and computes the evaluation measures.

        Args:
            y_true: True target labels.
            y_pred: Predicted target labels.
            X: Array/matrix of observations.
            predictor: Predictor object.
            rng: A numpy random number generator object.

        Raises:
            TypeError: If the calculation of a measure runs an error.
        """
        self.kwargs['y_true'] = y_true
        self.kwargs['y_pred'] = y_pred
        self.kwargs['X'] = X
        self.kwargs['predictor'] = predictor
        self.kwargs['rng'] = rng
        self.kwargs['result'] = self.result

        for measure_func in self.measure_funcs:  # run each evaluation measure
            try:
                # Get relevant keyword arguments
                call_args = dict()
                for arg in inspect.signature(measure_func).parameters.values():
                    if arg.name in self.kwargs.keys():
                        call_args[arg.name] = self.kwargs[arg.name]

                        if arg.name == 'reference_measure':
                            # For some measures, we require a reference performance metric.
                            # Accordingly, we also need to provide the arguments of that reference metric.
                            ref_meas_call_args = dict()
                            for ref_meas_arg in inspect.signature(call_args[arg.name]).parameters.values():
                                if ref_meas_arg.name in self.kwargs.keys() and ref_meas_arg.name not in ['y_true', 'y_pred']:
                                    ref_meas_call_args[ref_meas_arg.name] = self.kwargs[ref_meas_arg.name]
                            call_args['reference_measure_kwargs'] = ref_meas_call_args

                # Make function call and save measurement
                new_measure_val = measure_func(**call_args)
                measure_name = type(self.kwargs['metric']).__name__.lower() \
                    if measure_func.__name__ == 'river_classification_metric' else measure_func.__name__
                self.result[measure_name]['measures'].append(new_measure_val)
                self.result[measure_name]['mean'].append(np.nanmean(self.result[measure_name]['measures']))
                self.result[measure_name]['var'].append(np.nanvar(self.result[measure_name]['measures']))

                if self.decay_rate:
                    if len(self.result[measure_name]['mean_decay']) > 0 \
                            and not np.isnan(self.result[measure_name]['mean_decay'][-1]):
                        delta = new_measure_val - self.result[measure_name]['mean_decay'][-1]
                        self.result[measure_name]['mean_decay'].append(
                            self.result[measure_name]['mean_decay'][-1] + self.decay_rate * delta
                        )
                        self.result[measure_name]['var_decay'].append(
                            (1 - self.decay_rate) * (
                                        self.result[measure_name]['var_decay'][-1] + self.decay_rate * delta ** 2)
                        )
                    else:
                        self.result[measure_name]['mean_decay'].append(new_measure_val)
                        self.result[measure_name]['var_decay'].append(.0)

                if self.window_size:
                    self.result[measure_name]['mean_window'].append(
                        np.nanmean(self.result[measure_name]['measures'][-self.window_size:])
                    )
                    self.result[measure_name]['var_window'].append(
                        np.nanvar(self.result[measure_name]['measures'][-self.window_size:])
                    )
            except TypeError:
                traceback.print_exc()
                continue

    @staticmethod
    def _validate_func(func: Callable, kwargs: dict):
        """Validates the provided metric function.

        Args:
            func: Evaluation/metric function
            kwargs:
                A dictionary containing additional and specific keyword arguments, which are passed to the evaluation
                function.

        Raises:
            TypeError: If an invalid metric function was provided.
            AttributeError: If a non-keyword argument is missing from the provided parameters.
        """
        if not callable(func):
            raise TypeError("Please provide a valid metric function.")

        param_list = list(kwargs.keys())
        param_list.extend(['y_true', 'y_pred', 'X', 'predictor', 'result'])  # Arguments will be passed directly by the evaluator

        for arg in inspect.signature(func).parameters.values():
            if arg.default is arg.empty and arg.name not in param_list and arg.name != 'kwargs':
                raise AttributeError("The keyword argument '{}' of the evaluation measure '{}' has not been provided. "
                                     "Please provide the parameter in the constructor of the PredictionEvaluator object "
                                     "or use another evaluation measure.".format(arg.name, func.__name__, arg.name))
