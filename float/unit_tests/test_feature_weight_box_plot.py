import float.visualization as fvis
import matplotlib.pyplot as plt
import numpy as np
import unittest


class TestFeatureWeightBoxPlot(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        self.sample_feature_weights = [[list(np.random.randint(0, 1, 50)) for _ in range(100)] for _ in range(3)]

    def test_feature_weight_box_plot(self):
        ax = fvis.feature_weight_box_plot(feature_weights=self.sample_feature_weights,
                                          model_names=[f'Model {i}' for i in range(3)],
                                          feature_names=[f'Feature {i}' for i in range(50)],
                                          top_n_features=10)
        self.assertIsInstance(ax, plt.Axes, msg='feature_weight_box_plot() returns an object of type Axes')
        self.assertEqual(len(ax.artists), len(self.sample_feature_weights) * 10, msg='feature_weight_box_plot() draws the correct amount of boxes')
        self.assertEqual(ax.get_xlabel(), 'Input Feature', msg='feature_weight_box_plot() sets the correct xlabel')
        self.assertEqual(ax.get_ylabel(), 'Feature Weights', msg='feature_weight_box_plot() sets the correct ylabel')
        self.assertEqual([text.get_text() for text in ax.get_legend().texts], [f'Model {i}' for i in range(3)], msg='feature_weight_box_plot() sets the right legend texts')
