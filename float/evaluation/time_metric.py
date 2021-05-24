import numpy as np
from float.evaluation import Evaluator


class TimeMetric(Evaluator):
    def __init__(self):
        super().__init__(line_plot=True, scatter_plot=True, bar_plot=True)
        self.mean = None
        self.var = None

    def compute(self, start, end):
        self.measures.extend([end - start])
        self.mean = np.mean(self.measures)
        self.var = np.var(self.measures)
