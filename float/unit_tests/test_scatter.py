import float.visualization as fvis
import matplotlib.pyplot as plt
import numpy as np
import unittest


class TestScatter(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        self.sample_measures = [list(np.random.rand(200)) for _ in range(3)]

    def test_scatter(self):
        ax = fvis.plot(measures=self.sample_measures, legend_labels=[f'Model {i}' for i in range(3)], y_label='Measure')
        self.assertIsInstance(ax, plt.Axes, msg='scatter() returns an object of type Axes')
        self.assertEqual(len(ax.lines), len(self.sample_measures), msg='scatter() draws the correct amount of plots')
        self.assertEqual(ax.get_xlabel(), 'Time Step $t$', msg='scatter() sets the correct xlabel')
        self.assertEqual(ax.get_ylabel(), 'Measure', msg='scatter() sets the correct ylabel')
        self.assertEqual([text.get_text() for text in ax.get_legend().texts], [f'Model {i}' for i in range(3)], msg='scatter() sets the right legend texts')
