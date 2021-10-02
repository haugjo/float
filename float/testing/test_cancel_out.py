import unittest
import numpy as np
from float.data.data_loader import DataLoader
from float.feature_selection.evaluation.measures import nogueira_stability
from float.feature_selection.cancel_out import CancelOutFeatureSelector
from float.feature_selection.evaluation.feature_selection_evaluator import FeatureSelectionEvaluator


class TestCancelOutFeatureSelector(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        self.data_loader = DataLoader(file_path='../data/datasets/spambase.csv', target_col=0)
        self.ref_sample, _ = self.data_loader.stream.next_sample(50)
        self.cancel_out = CancelOutFeatureSelector(self.data_loader.stream.n_features, 10, reset_after_drift=False,
                                                   baseline='constant', ref_sample=self.ref_sample)
        self.fs_evaluator = FeatureSelectionEvaluator([nogueira_stability])

    def test_init(self):
        self.assertEqual(self.cancel_out.reset_after_drift, False, msg='attribute reset_after_drift is initialized correctly')
        self.assertIsInstance(self.cancel_out.n_total_features, int, msg='attribute n_total_features is initialized correctly')
        self.assertIsInstance(self.cancel_out.n_selected_features, int, msg='attribute n_selected_features is initialized correctly')
        self.assertIsInstance(self.cancel_out.baseline, str, msg='attribute baseline is initialized correctly')
        self.assertTrue(type(self.cancel_out.ref_sample) in [float, np.ndarray], msg='attribute ref_sample is initialized correctly')

        self.assertEqual(self.cancel_out.supports_multi_class, False, msg='attribute supports_multi_class is initialized correctly')

        self.assertEqual(self.cancel_out.raw_weight_vector.shape, (self.cancel_out.n_total_features, ), msg='attribute raw_weight_vector is initialized correctly')
        self.assertIsInstance(self.cancel_out.weights, list)
        self.assertIsInstance(self.cancel_out.selection, list)
        self.assertIsInstance(self.cancel_out.selected_features, list)
        self.assertEqual(self.cancel_out._auto_scale, False, msg='attribute auto_scale is initialized correctly')

    def test_weight_features(self):
        raw_weight_vector = self.cancel_out.raw_weight_vector
        X, y = self.data_loader.get_data(50)
        self.cancel_out.weight_features(X, y)
        self.assertEqual(raw_weight_vector.shape, self.cancel_out.raw_weight_vector.shape, msg='weight_features() preserves the shape of attribute raw_weight_vector')
        self.assertFalse((raw_weight_vector == self.cancel_out.raw_weight_vector).all(), msg='weight_features() updates attribute raw_weight_vector')

    def test_select_features(self):
        X, _ = self.data_loader.get_data(50)
        X_new = self.cancel_out.select_features(X)
        self.assertEqual(len(self.cancel_out.selected_features), self.cancel_out.n_selected_features, msg='select_features() sets the attribute selected_features correctly')
        self.assertEqual(len(self.cancel_out.weights), 1, msg='select_features() sets the attribute weights correctly')
        self.assertEqual(len(self.cancel_out.weights[0]), self.cancel_out.n_total_features, msg='select_features() sets the attribute weights correctly')
        self.assertEqual(len(self.cancel_out.selection), 1, msg='select_features() sets the attribute selection correctly')
        self.assertEqual(len(self.cancel_out.selection[0]), self.cancel_out.n_selected_features, msg='select_features() sets the attribute selection correctly')
        self.assertTrue((X_new[:, ~self.cancel_out.selected_features] == 0).all(), msg='select_features() sets the non-selected features correctly to 0')
        self.assertEqual(X.shape, X_new.shape, msg='select_features() preserves the shape of the input array')
        self.assertFalse((X == X_new).all(), msg='select_features() updates the input array')
