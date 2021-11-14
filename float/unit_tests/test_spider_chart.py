import float.visualization as fvis
import matplotlib.pyplot as plt
import numpy as np
import unittest


class TestSpiderChart(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        self.sample_measures = [list(np.random.rand(4)) for _ in range(3)]

    def test_spider_chart(self):
        ax = fvis.spider_chart(measures=self.sample_measures,
                               metric_names=[f'metric_{i}' for i in range(4)],
                               legend_names=[f'Model {i}' for i in range(3)],
                               ranges=[None, None, (0, 2), (0, 2)],
                               invert=[False, True, False, True])
        self.assertIsInstance(ax, plt.PolarAxes, msg='spider_chart() returns an object of type PolarAxes')
        self.assertEqual(len(ax.lines), 3, msg='spider_chart() draws the correct amount of plots')
        self.assertEqual(len(ax.patches), 3, msg='spider_chart() draws the correct amount of plots')
        self.assertEqual([text.get_text() for text in ax.axes.texts], [f'metric_{i}' for i in range(4)], msg='spider_chart() sets the correct xticklabels (metric names)')
        self.assertEqual([text.get_text() for text in ax.get_yticklabels()], ['', '0.2', '0.4', '0.6', '0.8', '1.0'], msg='spider_chart() sets the correct yticklabels)')
        self.assertEqual([text.get_text() for text in ax.get_legend().texts], [f'Model {i}' for i in range(3)], msg='spider_chart() sets the right legend texts')
        self.assertEqual([self.sample_measures[0][0], 1 - self.sample_measures[0][1], self.sample_measures[0][2] / 2, 1 - (self.sample_measures[0][3] / 2)], [x[1] for x in ax.patches[0].xy][:-1], msg='spider_chart() takes ranges and invert correctly into account')
