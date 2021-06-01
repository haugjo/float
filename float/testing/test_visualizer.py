import unittest
from matplotlib.axes import Axes
from float.visualization.visualizer import Visualizer


# TODO update (after updating Visualizer)
class TestVisualizer(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        measures = []
        self.visualizer = Visualizer(measures, fig_size=(20, 10))

    def test_init(self):
        self.assertIsInstance(self.visualizer.measures, list,
                              msg='attribute measures is initialized correctly')
        self.assertEqual(self.visualizer.fig_size, (20, 10), msg='attribute fig_size is initialized correctly')

    def test_plot(self):
        self.assertIsInstance(self.visualizer.plot(), Axes,
                              msg='plot() returns Axes object')

    def test_scatter(self):
        self.assertIsInstance(self.visualizer.scatter(), Axes,
                              msg='scatter() returns Axes object')

    def test_bar(self):
        self.assertIsInstance(self.visualizer.bar(), Axes,
                              msg='bar() returns Axes object')
