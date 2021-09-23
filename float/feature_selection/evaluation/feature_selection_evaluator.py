from abc import ABCMeta
import traceback
import numpy as np


class FeatureSelectionEvaluator(metaclass=ABCMeta):
    """
    Base class for the feature selection evaluation

    Attributes:
        measures (list): list of evaluation measure functions
        result (dict): dictionary of results per evaluation measure
    """
    def __init__(self, measures, decay_rate=None, window_size=None):
        """ Initialize feature selection evaluation measure

        Args:
            measures (list): list of evaluation measure functions
            decay_rate (float | None): when this parameter is set, the metric values are additionally aggregated with a decay/fading factor
            window_size (int | None): when this parameter is set, the metric values are additionally aggregated in a sliding window
        """
        self.decay_rate = decay_rate
        self.window_size = window_size

        self.measures = measures

        self.result = dict()
        for measure in measures:
            self.result[measure.__name__] = dict()
            self.result[measure.__name__]['measures'] = []

            if self.decay_rate:
                self.result[measure.__name__]['mean_decay'] = []
                self.result[measure.__name__]['var_decay'] = []

            if self.window_size:
                self.result[measure.__name__]['mean_window'] = []
                self.result[measure.__name__]['var_window'] = []

    def run(self, selected_features, n_total_features):
        """
        Compute and save each evaluation measure

        Args:
            selected_features (list): vector of selected features per time step
            n_total_features (int): total number of features
        """
        for measure in self.measures:  # run each evaluation measure
            try:
                new_measure = measure(selected_features, n_total_features)
                self.result[measure.__name__]['measures'].append(new_measure)
                self.result[measure.__name__]['mean'] = np.mean(self.result[measure.__name__]['measures'])
                self.result[measure.__name__]['var'] = np.var(self.result[measure.__name__]['measures'])

                if self.decay_rate:
                    if len(self.result[measure.__name__]['mean_decay']) > 0:
                        delta = new_measure - self.result[measure.__name__]['mean_decay'][-1]
                        self.result[measure.__name__]['mean_decay'].append(
                            self.result[measure.__name__]['mean_decay'][-1] + self.decay_rate * delta
                        )
                        self.result[measure.__name__]['var_decay'].append(
                            (1 - self.decay_rate) * (self.result[measure.__name__]['var_decay'][-1] + self.decay_rate * delta ** 2)
                        )
                    else:
                        self.result[measure.__name__]['mean_decay'].append(new_measure)
                        self.result[measure.__name__]['var_decay'].append(.0)

                if self.window_size:
                    self.result[measure.__name__]['mean_window'].append(
                        np.mean(self.result[measure.__name__]['measures'][-self.window_size:])
                    )
                    self.result[measure.__name__]['var_window'].append(
                        np.var(self.result[measure.__name__]['measures'][-self.window_size:])
                    )
            except TypeError:
                traceback.print_exc()
                continue
