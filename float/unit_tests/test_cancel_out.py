import unittest
import numpy as np
from float.data.data_loader import DataLoader
from float.feature_selection.cancel_out import CancelOutFeatureSelector


class TestCancelOutFeatureSelector(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        self.rng = np.random.default_rng(seed=0)
        self.data_loader = DataLoader(path='../data/datasets/spambase.csv', target_col=-1)
        self.ref_sample, _ = self.data_loader.stream.next_sample(10)
        self.cancel_out = CancelOutFeatureSelector(n_total_features=self.data_loader.stream.n_features,
                                                   n_selected_features=10,
                                                   baseline='constant',
                                                   ref_sample=self.ref_sample)

    def test_init(self):
        self.assertEqual(self.cancel_out.reset_after_drift, False, msg='attribute reset_after_drift is initialized correctly')
        self.assertIsInstance(self.cancel_out.n_total_features, int, msg='attribute n_total_features is initialized correctly')
        self.assertIsInstance(self.cancel_out.n_selected_features, int, msg='attribute n_selected_features is initialized correctly')
        self.assertIsInstance(self.cancel_out.baseline, str, msg='attribute baseline is initialized correctly')
        self.assertTrue(type(self.cancel_out.ref_sample) in [float, np.ndarray], msg='attribute ref_sample is initialized correctly')

        self.assertEqual(self.cancel_out.supports_multi_class, False, msg='attribute supports_multi_class is initialized correctly')

        self.assertEqual(self.cancel_out.weights.shape, (self.cancel_out.n_total_features,), msg='attribute weights is initialized correctly')
        self.assertIsInstance(self.cancel_out.weights_history, list, msg='attribute weights_history is initialized correctly')
        self.assertIsInstance(self.cancel_out.selected_features_history, list, msg='attribute selected_features_history is initialized correctly')
        self.assertIsInstance(self.cancel_out.selected_features, list, msg='attribute selected_features is initialized correctly')
        self.assertEqual(self.cancel_out._scale_warning_issued, False, msg='attribute _scale_warning_issued is initialized correctly')

    def test_weight_features(self):
        weights = self.cancel_out.weights
        X, y = self.data_loader.get_data(10)
        self.cancel_out.weight_features(X, y)
        self.assertEqual(weights.shape, self.cancel_out.weights.shape, msg='weight_features() preserves the shape of attribute weights')
        self.assertFalse((weights == self.cancel_out.weights).all(), msg='weight_features() updates attribute weights')

    def test_select_features(self):
        X, _ = self.data_loader.get_data(10)
        X_new = self.cancel_out.select_features(X, self.rng)
        self.assertEqual(len(self.cancel_out.selected_features), self.cancel_out.n_selected_features, msg='select_features() sets the attribute selected_features correctly')
        self.assertEqual(len(self.cancel_out.weights_history), 1, msg='select_features() sets the attribute weights_history correctly')
        self.assertEqual(len(self.cancel_out.weights_history[0]), self.cancel_out.n_total_features, msg='select_features() sets the attribute weights_history correctly')
        self.assertEqual(len(self.cancel_out.selected_features_history), 1, msg='select_features() sets the attribute selected_features_history correctly')
        self.assertEqual(len(self.cancel_out.selected_features_history[0]), self.cancel_out.n_selected_features, msg='select_features() sets the attribute selected_features_history correctly')
        self.assertTrue((X_new[:, ~self.cancel_out.selected_features] == 0).all(), msg='select_features() sets the non-selected features correctly to 0')
        self.assertEqual(X.shape, X_new.shape, msg='select_features() preserves the shape of the input array')
        self.assertFalse((X == X_new).all(), msg='select_features() updates the input array')
