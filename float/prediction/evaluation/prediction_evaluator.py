from abc import ABCMeta
import traceback
import numpy as np
import inspect
import copy
from float.prediction import BasePredictor


class PredictionEvaluator(metaclass=ABCMeta):
    """
    Abstract base class for prediction evaluation measures

    Attributes:
        measure_funcs (list): list of evaluation measure functions
        result (dict): dictionary of results per evaluation measure
        testing_times (list): testing times per time step
        training_times (list): training times per time step
    """

    def __init__(self, measure_funcs, decay_rate=None, window_size=None, **kwargs):
        """ Initialize change detection evaluation measure

        Args:
            measure_funcs (list): list of evaluation measure functions
            decay_rate (float | None): when this parameter is set, the metric values are additionally aggregated with a decay/fading factor
            window_size (int | None): when this parameter is set, the metric values are additionally aggregated in a sliding window
            kwargs: additional keyword arguments for the given measures
        """
        self.decay_rate = decay_rate
        self.window_size = window_size
        self.kwargs = kwargs
        self.measure_funcs = measure_funcs  # Todo: name measure_funcs

        self.testing_times = []
        self.training_times = []

        self.result = dict()
        for measure_func in measure_funcs:
            self._validate_func(measure_func, kwargs)

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

    def run(self, y_true, y_pred, X, predictor):
        """
        Compute and save each evaluation measure

        Args:
            y_true (list | np.array): true target label
            y_pred (list | np.array): predicted target label
            X (np.array): matrix of observations
            predictor (BasePredictor): predictor object
        """
        self.kwargs['y_true'] = copy.copy(y_true)
        self.kwargs['y_pred'] = copy.copy(y_pred)
        self.kwargs['X'] = copy.copy(X)
        self.kwargs['predictor'] = copy.deepcopy(predictor)
        self.kwargs['result'] = copy.deepcopy(self.result)

        for measure_func in self.measure_funcs:  # run each evaluation measure
            try:
                # Get relevant keyword arguments
                call_args = dict()
                for arg in inspect.signature(measure_func).parameters.values():
                    if arg.name in self.kwargs.keys():
                        call_args[arg.name] = self.kwargs[arg.name]

                # Make function call and save measurement
                new_measure_val = measure_func(**call_args)
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
                            (1 - self.decay_rate) * (
                                        self.result[measure_func.__name__]['var_decay'][-1] + self.decay_rate * delta ** 2)
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

    @staticmethod
    def _validate_func(func, kwargs):
        """
        Validate the provided metric function

        Args:
            func (function): evaluation/metric function
            kwargs (dict): additional keyword arguments for the given measures
        """
        if not callable(func):
            raise TypeError("Please provide a valid metric function.")

        param_list = list(kwargs.keys())
        param_list.extend(['y_true', 'y_pred', 'X', 'predictor', 'result'])  # Note: arguments will be provided by the evaluator

        for arg in inspect.signature(func).parameters.values():
            if arg.default is arg.empty and arg.name not in param_list:
                raise AttributeError("The non-keyword argument '{}' of the evaluation measure '{}' has not been provided. "
                                     "Please provide the parameter in the constructor of the PredictionEvaluator object "
                                     "or use another evaluation measure.".format(arg.name, func.__name__, arg.name))
