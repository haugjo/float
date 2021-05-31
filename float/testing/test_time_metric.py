import unittest
import time
from float.evaluation.time_metric import TimeMetric


class TestTimeMetric(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        self.time_metric = TimeMetric()

    def test_init(self):
        self.assertEqual(self.time_metric.line_plot, True, msg='attribute line_plot is set to True')
        self.assertEqual(self.time_metric.scatter_plot, True, msg='attribute scatter_plot is set to True')
        self.assertEqual(self.time_metric.bar_plot, True, msg='attribute bar_plot is set to True')

    def test_compute(self):
        len_measures = len(self.time_metric.measures)
        self.time_metric.compute(time.time(), time.time())
        self.assertEqual(len(self.time_metric.measures), len_measures + 1,
                         msg='compute() increases len of measures by 1')
        self.assertIsInstance(self.time_metric.mean, float, 'attribute mean is of type float after compute() is called')
        self.assertIsInstance(self.time_metric.var, float, 'attribute var is of type float after compute() is called')
