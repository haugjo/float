import float.visualization as fvis
import matplotlib.pyplot as plt
import numpy as np
import unittest


class TestFeatureSelectionBar(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        self.sample_selected_features = [[list(np.random.randint(0, 50, 20)) for _ in range(100)] for _ in range(3)]

    def test_feature_selection_bar(self):
        ax = fvis.feature_selection_bar(selected_features=self.sample_selected_features,
                                        model_names=[f'Model {i}' for i in range(3)],
                                        feature_names=[f'Feature {i}' for i in range(50)],
                                        top_n_features=10)
        self.assertIsInstance(ax, plt.Axes, msg='feature_selection_bar() returns an object of type Axes')
        self.assertEqual(len(ax.patches), len(self.sample_selected_features) * 10, msg='feature_selection_bar() draws the correct amount of bars')
        self.assertEqual(ax.get_xlabel(), 'Input Feature', msg='feature_selection_bar() sets the correct xlabel')
        self.assertEqual(ax.get_ylabel(), 'No. Times Selected', msg='feature_selection_bar() sets the correct ylabel')
        self.assertEqual([text.get_text() for text in ax.get_legend().texts], [f'Model {i}' for i in range(3)], msg='feature_selection_bar() sets the right legend texts')
