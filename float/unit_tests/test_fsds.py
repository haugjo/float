import unittest
from float.data.data_loader import DataLoader
from float.feature_selection.fsds import FSDS


class TestFSDS(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        self.data_loader = DataLoader(file_path='../data/datasets/spambase.csv', target_col=0)
        self.ref_sample, _ = self.data_loader.stream.next_sample(50)
        self.fsds = FSDS(self.data_loader.stream.n_features, 10, reset_after_drift=False,
                         baseline='constant', ref_sample=self.ref_sample)

    def test_init(self):
        self.assertEqual(self.fsds.reset_after_drift, False, msg='attribute reset_after_drift is initialized correctly')
        self.assertIsInstance(self.fsds.n_total_features, int, msg='attribute n_total_features is initialized correctly')
        self.assertIsInstance(self.fsds.n_selected_features, int, msg='attribute n_selected_features is initialized correctly')
        self.assertIsInstance(self.fsds.baseline, str, msg='attribute baseline is initialized correctly')
        self.assertTrue(type(self.fsds.ref_sample) in [float, np.ndarray], msg='attribute ref_sample is initialized correctly')

        self.assertEqual(self.fsds.supports_multi_class, False, msg='attribute supports_multi_class is initialized correctly')

        self.assertEqual(self.fsds.raw_weight_vector.shape, (self.fsds.n_total_features,), msg='attribute raw_weight_vector is initialized correctly')
        self.assertIsInstance(self.fsds.weights, list)
        self.assertIsInstance(self.fsds.selection, list)
        self.assertIsInstance(self.fsds.selected_features, list)
        self.assertEqual(self.fsds._auto_scale, False, msg='attribute auto_scale is initialized correctly')

    def test_weight_features(self):
        raw_weight_vector = self.fsds.raw_weight_vector
        X, y = self.data_loader.get_data(50)
        self.fsds.weight_features(X, y)
        self.assertEqual(raw_weight_vector.shape, self.fsds.raw_weight_vector.shape, msg='weight_features() preserves the shape of attribute raw_weight_vector')
        self.assertFalse((raw_weight_vector == self.fsds.raw_weight_vector).all(), msg='weight_features() updates attribute raw_weight_vector')

    def test_select_features(self):
        X, _ = self.data_loader.get_data(50)
        X_new = self.fsds.select_features(X)
        self.assertEqual(len(self.fsds.selected_features), self.fsds.n_selected_features, msg='select_features() sets the attribute selected_features correctly')
        self.assertEqual(len(self.fsds.weights), 1, msg='select_features() sets the attribute weights correctly')
        self.assertEqual(len(self.fsds.weights[0]), self.fsds.n_total_features, msg='select_features() sets the attribute weights correctly')
        self.assertEqual(len(self.fsds.selection), 1, msg='select_features() sets the attribute selection correctly')
        self.assertEqual(len(self.fsds.selection[0]), self.fsds.n_selected_features, msg='select_features() sets the attribute selection correctly')
        self.assertTrue((X_new[:, ~self.fsds.selected_features] == 0).all(), msg='select_features() sets the non-selected features correctly to 0')
        self.assertEqual(X.shape, X_new.shape, msg='select_features() preserves the shape of the input array')
        self.assertFalse((X == X_new).all(), msg='select_features() updates the input array')

    def test_reset(self):
        self.fsds.reset()
        self.assertEqual(self.fsds.m, self.fsds.n_total_features, msg='reset() resets attribute m correctly')
        self.assertEqual(self.fsds.B, [], msg='reset() resets attribute B correctly')
