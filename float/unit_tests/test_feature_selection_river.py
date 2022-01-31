import numpy as np
from river import stats
from river.feature_selection.k_best import SelectKBest
from river.feature_selection.random import PoissonInclusion
from river.feature_selection.variance import VarianceThreshold
import unittest

from float.data.data_loader import DataLoader
from float.feature_selection.river import RiverFeatureSelector


class TestRiverFeatureSelector(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        self.rng = np.random.default_rng(seed=0)
        self.data_loader = DataLoader(path='../data/datasets/spambase.csv', target_col=-1)
        self.ref_sample, _ = self.data_loader.stream.next_sample(10)
        self.feature_selectors = {'SelectKBest': SelectKBest(similarity=stats.PearsonCorr()),
                                  'VarianceThreshold': VarianceThreshold(),
                                  'PoissonInclusion': PoissonInclusion(p=0.5)}
        self.river_feature_selectors = []
        for fs in self.feature_selectors.values():
            self.river_feature_selectors.append(RiverFeatureSelector(model=fs,
                                                                     feature_names=self.data_loader.stream.feature_names,
                                                                     n_total_features=self.data_loader.stream.n_features,
                                                                     ref_sample=self.ref_sample))

    def test_init(self):
        for i, k in enumerate(self.feature_selectors.keys()):
            self.assertEqual(self.river_feature_selectors[i].reset_after_drift, False,
                             msg=f'attribute reset_after_drift is initialized correctly ({k})')
            self.assertIsInstance(self.river_feature_selectors[i].n_total_features, int,
                                  msg=f'attribute n_total_features is initialized correctly ({k})')
            self.assertIsInstance(self.river_feature_selectors[i].baseline, str,
                                  msg=f'attribute baseline is initialized correctly ({k})')
            self.assertTrue(type(self.river_feature_selectors[i].ref_sample) in [float, np.ndarray],
                            msg=f'attribute ref_sample is initialized correctly ({k})')
            self.assertFalse(self.river_feature_selectors[i].supports_multi_class,
                             msg=f'attribute supports_multi_class is initialized correctly ({k})')
            self.assertEqual(self.river_feature_selectors[i].weights.shape,
                             (self.river_feature_selectors[i].n_total_features,),
                             msg=f'attribute weights is initialized correctly ({k})')
            self.assertIsInstance(self.river_feature_selectors[i].weights_history, list,
                                  msg=f'attribute weights_history is initialized correctly ({k})')
            self.assertIsInstance(self.river_feature_selectors[i].selected_features_history, list,
                                  msg=f'attribute selected_features_history is initialized correctly ({k})')
            self.assertIsInstance(self.river_feature_selectors[i].selected_features, list,
                                  msg=f'attribute selected_features is initialized correctly ({k})')
            self.assertFalse(self.river_feature_selectors[i]._scale_warning_issued,
                             msg=f'attribute _scale_warning_issued is initialized correctly ({k})')

    def test_weight_features(self):
        for i, k in enumerate(self.feature_selectors.keys()):
            if k != 'PoissonInclusion':
                weights = self.river_feature_selectors[i].weights
                X, y = self.data_loader.get_data(10)
                self.river_feature_selectors[i].weight_features(X, y)
                self.assertEqual(weights.shape, self.river_feature_selectors[i].weights.shape,
                                 msg=f'weight_features() preserves the shape of attribute weights ({k})')
                self.assertFalse((weights == self.river_feature_selectors[i].weights).all(),
                                 msg=f'weight_features() updates attribute weights ({k})')

    def test_select_features(self):
        for i, k in enumerate(self.feature_selectors.keys()):
            X, _ = self.data_loader.get_data(10)
            X_new = self.river_feature_selectors[i].select_features(X, self.rng)

            if self.river_feature_selectors[i].n_selected_features:
                self.assertEqual(len(self.river_feature_selectors[i].selected_features),
                                 self.river_feature_selectors[i].n_selected_features,
                                 msg=f'select_features() sets the attribute selected_features correctly ({k})')
                self.assertEqual(len(self.river_feature_selectors[i].selected_features_history[0]),
                                 self.river_feature_selectors[i].n_selected_features,
                                 msg=f'select_features() sets the attribute selected_features_history correctly ({k})')

            self.assertEqual(len(self.river_feature_selectors[i].weights_history), 1,
                             msg=f'select_features() sets the attribute weights_history correctly ({k})')
            self.assertEqual(len(self.river_feature_selectors[i].weights_history[0]),
                             self.river_feature_selectors[i].n_total_features,
                             msg=f'select_features() sets the attribute weights_history correctly ({k})')
            self.assertEqual(len(self.river_feature_selectors[i].selected_features_history), 1,
                             msg=f'select_features() sets the attribute selected_features_history correctly ({k})')

            if len(self.river_feature_selectors[i].selected_features) \
                    < len(self.river_feature_selectors[i].feature_names):
                self.assertTrue((X_new[:, ~self.river_feature_selectors[i].selected_features] == 0).all(),
                                msg=f'select_features() sets the non-selected features correctly to 0 ({k})')
                self.assertEqual(X.shape, X_new.shape,
                                 msg=f'select_features() preserves the shape of the input array ({k})')
                self.assertFalse((X == X_new).all(),
                                 msg=f'select_features() updates the input array ({k})')

    def test_reset(self):
        for i, k in enumerate(self.feature_selectors.keys()):
            weights = self.river_feature_selectors[i].weights
            self.river_feature_selectors[i].reset()
            self.assertEqual(weights.shape, self.river_feature_selectors[i].weights.shape,
                             msg=f'reset() preserves the shape of attribute weights ({k})')
            self.assertTrue((self.river_feature_selectors[i].weights == 0).all(),
                            msg=f'reset() resets attribute weights correctly ({k})')
