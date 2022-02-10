from float.data.data_loader import DataLoader
from float.feature_selection.evaluation.feature_selection_evaluator import FeatureSelectionEvaluator
from float.feature_selection.evaluation.measures.nogueira_stability import nogueira_stability
from float.feature_selection.fires import FIRES
import numpy as np
import unittest


class TestFeatureSelectionEvaluator(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        self.rng = np.random.default_rng(seed=0)
        self.data_loader = DataLoader(path='../datasets/spambase.csv', target_col=-1)
        ref_sample, _ = self.data_loader.stream.next_sample(10)
        self.data_loader.stream.reset()
        self.fires = FIRES(n_total_features=self.data_loader.stream.n_features,
                           n_selected_features=20,
                           classes=self.data_loader.stream.target_values,
                           reset_after_drift=False,
                           baseline='expectation',
                           ref_sample=ref_sample)
        self.feature_selection_evaluator = FeatureSelectionEvaluator([nogueira_stability])

    def test_init(self):
        for measure_func in self.feature_selection_evaluator.measure_funcs:
            self.assertTrue(measure_func.__name__ in self.feature_selection_evaluator.result.keys())

    def test_run(self):
        X, y = self.data_loader.get_data(10)
        self.fires.weight_features(X, y)
        _ = self.fires.select_features(X, self.rng)
        self.feature_selection_evaluator.run(self.fires.selected_features_history, self.fires.n_total_features)
        self.assertTrue(len(self.feature_selection_evaluator.result[nogueira_stability.__name__]['measures']) > 0,
                        msg='run() adds a value to the measure dict')
        self.assertTrue(len(self.feature_selection_evaluator.result[nogueira_stability.__name__]['mean']) > 0,
                        msg='run() adds a value to the mean dict')
        self.assertTrue(len(self.feature_selection_evaluator.result[nogueira_stability.__name__]['var']) > 0,
                        msg='run() adds a value to the var dict')
