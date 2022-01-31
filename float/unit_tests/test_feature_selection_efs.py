from float.data.data_loader import DataLoader
from float.feature_selection.efs import EFS
import numpy as np
import unittest


class TestEFS(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        self.rng = np.random.default_rng(seed=0)
        self.data_loader = DataLoader(path='../data/datasets/spambase.csv', target_col=-1)
        self.ref_sample, _ = self.data_loader.stream.next_sample(10)
        self.efs = EFS(n_total_features=self.data_loader.stream.n_features,
                       n_selected_features=10,
                       baseline='constant',
                       ref_sample=self.ref_sample)

    def test_init(self):
        self.assertEqual(self.efs.reset_after_drift, False,
                         msg='attribute reset_after_drift is initialized correctly')
        self.assertIsInstance(self.efs.n_total_features, int, msg='attribute n_total_features is initialized correctly')
        self.assertIsInstance(self.efs.n_selected_features, int,
                              msg='attribute n_selected_features is initialized correctly')
        self.assertIsInstance(self.efs.baseline, str, msg='attribute baseline is initialized correctly')
        self.assertTrue(type(self.efs.ref_sample) in [float, np.ndarray],
                        msg='attribute ref_sample is initialized correctly')
        self.assertEqual(self.efs.supports_multi_class, False,
                         msg='attribute supports_multi_class is initialized correctly')
        self.assertEqual(self.efs.weights.shape, (self.efs.n_total_features,),
                         msg='attribute weights is initialized correctly')
        self.assertIsInstance(self.efs.weights_history, list, msg='attribute weights_history is initialized correctly')
        self.assertIsInstance(self.efs.selected_features_history, list,
                              msg='attribute selected_features_history is initialized correctly')
        self.assertIsInstance(self.efs.selected_features, list,
                              msg='attribute selected_features is initialized correctly')
        self.assertEqual(self.efs._scale_warning_issued, False,
                         msg='attribute _scale_warning_issued is initialized correctly')

    def test_weight_features(self):
        weights = self.efs.weights
        X, y = self.data_loader.get_data(10)
        self.efs.weight_features(X, y)
        self.assertEqual(weights.shape, self.efs.weights.shape,
                         msg='weight_features() preserves the shape of attribute weights')
        self.assertFalse((weights == self.efs.weights).all(), msg='weight_features() updates attribute weights')

    def test_select_features(self):
        X, _ = self.data_loader.get_data(10)
        X_new = self.efs.select_features(X, self.rng)
        self.assertEqual(len(self.efs.selected_features), self.efs.n_selected_features,
                         msg='select_features() sets the attribute selected_features correctly')
        self.assertEqual(len(self.efs.weights_history), 1,
                         msg='select_features() sets the attribute weights_history correctly')
        self.assertEqual(len(self.efs.weights_history[0]), self.efs.n_total_features,
                         msg='select_features() sets the attribute weights_history correctly')
        self.assertEqual(len(self.efs.selected_features_history), 1,
                         msg='select_features() sets the attribute selected_features_history correctly')
        self.assertEqual(len(self.efs.selected_features_history[0]), self.efs.n_selected_features,
                         msg='select_features() sets the attribute selected_features_history correctly')
        self.assertTrue((X_new[:, ~self.efs.selected_features] == 0).all(),
                        msg='select_features() sets the non-selected features correctly to 0')
        self.assertEqual(X.shape, X_new.shape, msg='select_features() preserves the shape of the input array')
        self.assertFalse((X == X_new).all(), msg='select_features() updates the input array')

    def test_reset(self):
        u, v = self.efs._u, self.efs._v
        self.efs.reset()
        self.assertEqual(u.shape, self.efs._u.shape, msg='reset() preserves the shape of attribute u')
        self.assertEqual(v.shape, self.efs._v.shape, msg='reset() preserves the shape of attribute v')
        self.assertTrue((self.efs._u == 2).all(), msg='reset() resets attribute u correctly')
        self.assertTrue((self.efs._v == 1).all(), msg='reset() resets attribute v correctly')
