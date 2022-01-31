from float.data.data_loader import DataLoader
from float.feature_selection.fsds import FSDS
import numpy as np
import unittest


class TestFSDS(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        self.rng = np.random.default_rng(seed=0)
        self.data_loader = DataLoader(path='../data/datasets/spambase.csv', target_col=-1)
        self.ref_sample, _ = self.data_loader.stream.next_sample(10)
        self.fsds = FSDS(n_total_features=self.data_loader.stream.n_features,
                         n_selected_features=10,
                         baseline='constant',
                         ref_sample=self.ref_sample)

    def test_init(self):
        self.assertEqual(self.fsds.reset_after_drift, False,
                         msg='attribute reset_after_drift is initialized correctly')
        self.assertIsInstance(self.fsds.n_total_features, int,
                              msg='attribute n_total_features is initialized correctly')
        self.assertIsInstance(self.fsds.n_selected_features, int,
                              msg='attribute n_selected_features is initialized correctly')
        self.assertIsInstance(self.fsds.baseline, str,
                              msg='attribute baseline is initialized correctly')
        self.assertTrue(type(self.fsds.ref_sample) in [float, np.ndarray],
                        msg='attribute ref_sample is initialized correctly')
        self.assertEqual(self.fsds.supports_multi_class, True,
                         msg='attribute supports_multi_class is initialized correctly')
        self.assertEqual(self.fsds.weights.shape, (self.fsds.n_total_features,),
                         msg='attribute weights is initialized correctly')
        self.assertIsInstance(self.fsds.weights_history, list,
                              msg='attribute weights_history is initialized correctly')
        self.assertIsInstance(self.fsds.selected_features_history, list,
                              msg='attribute selected_features_history is initialized correctly')
        self.assertIsInstance(self.fsds.selected_features, list,
                              msg='attribute selected_features is initialized correctly')
        self.assertEqual(self.fsds._scale_warning_issued, False,
                         msg='attribute _scale_warning_issued is initialized correctly')

    def test_weight_features(self):
        weights = self.fsds.weights
        X, y = self.data_loader.get_data(10)
        self.fsds.weight_features(X, y)
        self.assertEqual(weights.shape, self.fsds.weights.shape,
                         msg='weight_features() preserves the shape of attribute weights')
        self.assertFalse((weights == self.fsds.weights).all(), msg='weight_features() updates attribute weights')

    def test_select_features(self):
        X, _ = self.data_loader.get_data(10)
        X_new = self.fsds.select_features(X, self.rng)
        self.assertEqual(len(self.fsds.selected_features), self.fsds.n_selected_features,
                         msg='select_features() sets the attribute selected_features correctly')
        self.assertEqual(len(self.fsds.weights_history), 1,
                         msg='select_features() sets the attribute weights_history correctly')
        self.assertEqual(len(self.fsds.weights_history[0]), self.fsds.n_total_features,
                         msg='select_features() sets the attribute weights_history correctly')
        self.assertEqual(len(self.fsds.selected_features_history), 1,
                         msg='select_features() sets the attribute selected_features_history correctly')
        self.assertEqual(len(self.fsds.selected_features_history[0]), self.fsds.n_selected_features,
                         msg='select_features() sets the attribute selected_features_history correctly')
        self.assertTrue((X_new[:, ~self.fsds.selected_features] == 0).all(),
                        msg='select_features() sets the non-selected features correctly to 0')
        self.assertEqual(X.shape, X_new.shape, msg='select_features() preserves the shape of the input array')
        self.assertFalse((X == X_new).all(), msg='select_features() updates the input array')

    def test_reset(self):
        self.fsds.reset()
        self.assertEqual(self.fsds._m, self.fsds.n_total_features, msg='reset() resets attribute m correctly')
        self.assertEqual(self.fsds._B, [], msg='reset() resets attribute B correctly')
