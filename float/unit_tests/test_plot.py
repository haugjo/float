import float.visualization as fvis
import matplotlib.pyplot as plt
import numpy as np
import unittest


class TestPlot(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        self.sample_measures = [list(np.random.rand(200)) for _ in range(3)]

    def test_plot(self):
        ax = fvis.plot(measures=self.sample_measures, legend_labels=[f'Model {i}' for i in range(3)], y_label='Measure')
        self.assertIsInstance(ax, plt.Axes, msg='plot() returns an object of type Axes')
        self.assertEqual(len(ax.lines), len(self.sample_measures), msg='plot() draws the correct amount of plots')
        self.assertEqual(ax.get_xlabel(), 'Time Step $t$', msg='plot() sets the correct xlabel')
        self.assertEqual(ax.get_ylabel(), 'Measure', msg='plot() sets the correct ylabel')
        self.assertEqual([text.get_text() for text in ax.get_legend().texts], [f'Model {i}' for i in range(3)], msg='plot() sets the right legend texts')
