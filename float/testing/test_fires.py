import unittest
import numpy as np
from float.data.data_loader import DataLoader
from float.feature_selection.fires import FIRES


class TestFIRES(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        self.data_loader = DataLoader(file_path='../data/datasets/spambase.csv', target_col=0)
        self.ref_sample, _ = self.data_loader.stream.next_sample(50)
        self.fires = FIRES(n_total_features=self.data_loader.stream.n_features,
                           n_selected_features=20,
                           classes=self.data_loader.stream.target_values,
                           reset_after_drift=False,
                           baseline='constant',
                           ref_sample=self.ref_sample)

    def test_init(self):
        self.assertEqual(self.fires.reset_after_drift, False, msg='attribute reset_after_drift is initialized correctly')
        self.assertIsInstance(self.fires.n_total_features, int, msg='attribute n_total_features is initialized correctly')
        self.assertIsInstance(self.fires.n_selected_features, int, msg='attribute n_selected_features is initialized correctly')
        self.assertIsInstance(self.fires.baseline, str, msg='attribute baseline is initialized correctly')
        self.assertTrue(type(self.fires.ref_sample) in [float, np.ndarray], msg='attribute ref_sample is initialized correctly')

        self.assertEqual(self.fires.supports_multi_class, False, msg='attribute supports_multi_class is initialized correctly')

        self.assertEqual(self.fires.raw_weight_vector.shape, (self.fires.n_total_features,), msg='attribute raw_weight_vector is initialized correctly')
        self.assertIsInstance(self.fires.weights, list)
        self.assertIsInstance(self.fires.selection, list)
        self.assertIsInstance(self.fires.selected_features, list)
        self.assertEqual(self.fires._auto_scale, False, msg='attribute auto_scale is initialized correctly')

    def test_weight_features(self):
        raw_weight_vector = self.fires.raw_weight_vector
        X, y = self.data_loader.get_data(50)
        self.fires.weight_features(X, y)
        self.assertEqual(raw_weight_vector.shape, self.fires.raw_weight_vector.shape, msg='weight_features() preserves the shape of attribute raw_weight_vector')
        self.assertFalse((raw_weight_vector == self.fires.raw_weight_vector).all(), msg='weight_features() updates attribute raw_weight_vector')

    def test_select_features(self):
        X, _ = self.data_loader.get_data(50)
        X_new = self.fires.select_features(X)
        self.assertEqual(len(self.fires.selected_features), self.fires.n_selected_features, msg='select_features() sets the attribute selected_features correctly')
        self.assertEqual(len(self.fires.weights), 1, msg='select_features() sets the attribute weights correctly')
        self.assertEqual(len(self.fires.weights[0]), self.fires.n_total_features, msg='select_features() sets the attribute weights correctly')
        self.assertEqual(len(self.fires.selection), 1, msg='select_features() sets the attribute selection correctly')
        self.assertEqual(len(self.fires.selection[0]), self.fires.n_selected_features, msg='select_features() sets the attribute selection correctly')
        self.assertTrue((X_new[:, ~self.fires.selected_features] == 0).all(), msg='select_features() sets the non-selected features correctly to 0')
        self.assertEqual(X.shape, X_new.shape, msg='select_features() preserves the shape of the input array')
        self.assertFalse((X == X_new).all(), msg='select_features() updates the input array')

    def test_reset(self):
        mu, sigma = self.fires.mu, self.fires.sigma
        self.fires.reset()
        self.assertEqual(mu.shape, self.fires.mu.shape, msg='reset() preserves the shape of attribute mu')
        self.assertEqual(sigma.shape, self.fires.sigma.shape, msg='reset() preserves the shape of attribute sigma')
        self.assertTrue((self.fires.mu == (1 * self.fires.mu_init)).all(), msg='reset() resets attribute mu correctly')
        self.assertTrue((self.fires.sigma == (1 * self.fires.sigma_init)).all(), msg='reset() resets attribute sigma correctly')
