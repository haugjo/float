import float.visualization as fvis
import matplotlib.pyplot as plt
import numpy as np
import unittest


class TestFeatureSelectionScatter(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        self.sample_selected_features = [list(np.random.randint(0, 50, 20)) for _ in range(100)]

    def test_feature_selection_scatter(self):
        ax = fvis.feature_selection_scatter(selected_features=self.sample_selected_features)
        self.assertIsInstance(ax, plt.Axes, msg='feature_selection_scatter() returns an object of type Axes')
        self.assertEqual(ax.get_xlabel(), 'Time Step $t$', msg='feature_selection_scatter() sets the correct xlabel')
        self.assertEqual(ax.get_ylabel(), 'Feature Index', msg='feature_selection_scatter() sets the correct ylabel')
        self.assertEqual(ax.get_legend().texts[0].get_text(), 'Selected Feature Indicator',
                         msg='feature_selection_scatter() sets the right legend text')
