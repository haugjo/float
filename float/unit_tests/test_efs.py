import unittest
import numpy as np
from float.data.data_loader import DataLoader
from float.feature_selection.efs import EFS


class TestEFS(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        self.data_loader = DataLoader(file_path='../data/datasets/spambase.csv', target_col=0)
        self.ref_sample, _ = self.data_loader.stream.next_sample(50)
        self.efs = EFS(self.data_loader.stream.n_features, 10, reset_after_drift=False,
                       baseline='constant', ref_sample=self.ref_sample)

    def test_init(self):
        self.assertEqual(self.efs.reset_after_drift, False, msg='attribute reset_after_drift is initialized correctly')
        self.assertIsInstance(self.efs.n_total_features, int, msg='attribute n_total_features is initialized correctly')
        self.assertIsInstance(self.efs.n_selected_features, int, msg='attribute n_selected_features is initialized correctly')
        self.assertIsInstance(self.efs.baseline, str, msg='attribute baseline is initialized correctly')
        self.assertTrue(type(self.efs.ref_sample) in [float, np.ndarray], msg='attribute ref_sample is initialized correctly')

        self.assertEqual(self.efs.supports_multi_class, False, msg='attribute supports_multi_class is initialized correctly')

        self.assertEqual(self.efs.raw_weight_vector.shape, (self.efs.n_total_features,), msg='attribute raw_weight_vector is initialized correctly')
        self.assertIsInstance(self.efs.weights, list)
        self.assertIsInstance(self.efs.selection, list)
        self.assertIsInstance(self.efs.selected_features, list)
        self.assertEqual(self.efs._auto_scale, False, msg='attribute auto_scale is initialized correctly')

    def test_weight_features(self):
        raw_weight_vector = self.efs.raw_weight_vector
        X, y = self.data_loader.get_data(50)
        self.efs.weight_features(X, y)
        self.assertEqual(raw_weight_vector.shape, self.efs.raw_weight_vector.shape, msg='weight_features() preserves the shape of attribute raw_weight_vector')
        self.assertFalse((raw_weight_vector == self.efs.raw_weight_vector).all(), msg='weight_features() updates attribute raw_weight_vector')

    def test_select_features(self):
        X, _ = self.data_loader.get_data(50)
        X_new = self.efs.select_features(X)
        self.assertEqual(len(self.efs.selected_features), self.efs.n_selected_features, msg='select_features() sets the attribute selected_features correctly')
        self.assertEqual(len(self.efs.weights), 1, msg='select_features() sets the attribute weights correctly')
        self.assertEqual(len(self.efs.weights[0]), self.efs.n_total_features, msg='select_features() sets the attribute weights correctly')
        self.assertEqual(len(self.efs.selection), 1, msg='select_features() sets the attribute selection correctly')
        self.assertEqual(len(self.efs.selection[0]), self.efs.n_selected_features, msg='select_features() sets the attribute selection correctly')
        self.assertTrue((X_new[:, ~self.efs.selected_features] == 0).all(), msg='select_features() sets the non-selected features correctly to 0')
        self.assertEqual(X.shape, X_new.shape, msg='select_features() preserves the shape of the input array')
        self.assertFalse((X == X_new).all(), msg='select_features() updates the input array')

    def test_reset(self):
        u, v = self.efs.u, self.efs.v
        self.efs.reset()
        self.assertEqual(u.shape, self.efs.u.shape, msg='reset() preserves the shape of attribute u')
        self.assertEqual(v.shape, self.efs.v.shape, msg='reset() preserves the shape of attribute v')
        self.assertTrue((self.efs.u == 2).all(), msg='reset() resets attribute u correctly')
        self.assertTrue((self.efs.v == 1).all(), msg='reset() resets attribute v correctly')
