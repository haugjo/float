import unittest
from matplotlib.axes import Axes
from float.evaluation.evaluator import Evaluator
from float.evaluation.time_metric import TimeMetric
from float.visualization.visualizer import Visualizer


class TestVisualizer(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        time_metric = TimeMetric()
        self.visualizer = Visualizer(evaluator=time_metric, fig_size=(20, 10))

    def test_init(self):
        self.assertIsInstance(self.visualizer.evaluator, Evaluator,
                              msg='attribute evaluator is initialized correctly')
        self.assertEqual(self.visualizer.fig_size, (20,10), msg='attribute fig_size is initialized correctly')

    def test_plot(self):
        self.assertIsInstance(self.visualizer.plot(), Axes,
                              msg='plot() returns Axes object')
        self.visualizer.evaluator.line_plot = False
        self.assertIsNone(self.visualizer.plot(), msg='plot() returns None when Evaluator.line_plot is False')
        with self.assertWarns(Warning, msg='plot() triggers warning when Evaluator.line_plot is False'):
            self.visualizer.plot()

    def test_scatter(self):
        self.assertIsInstance(self.visualizer.scatter(), Axes,
                              msg='scatter() returns Axes object')
        self.visualizer.evaluator.scatter_plot = False
        self.assertIsNone(self.visualizer.scatter(), msg='scatter() returns None when Evaluator.scatter_plot is False')
        with self.assertWarns(Warning, msg='scatter() triggers warning when Evaluator.scatter_plot is False'):
            self.visualizer.scatter()

    def test_bar(self):
        self.assertIsInstance(self.visualizer.bar(), Axes,
                              msg='bar() returns Axes object')
        self.visualizer.evaluator.bar_plot = False
        self.assertIsNone(self.visualizer.bar(), msg='bar() returns None when Evaluator.bar_plot is False')
        with self.assertWarns(Warning, msg='bar() triggers warning when Evaluator.bar_plot is False'):
            self.visualizer.bar()
