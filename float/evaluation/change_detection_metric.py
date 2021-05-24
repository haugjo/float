from float.evaluation.evaluator import Evaluator


class ChangeDetectionMetric(Evaluator):
    def __init__(self):
        super().__init__(line_plot=True, scatter_plot=True, bar_plot=True)
        self.n_changes = 0

    def compute(self, change_detected):
        self.measures.append(change_detected)
        self.n_changes += change_detected
