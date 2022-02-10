from float.change_detection.erics import ERICS
from float.change_detection.evaluation.change_detection_evaluator import ChangeDetectionEvaluator
from float.change_detection.evaluation.measures import detection_delay, detected_change_rate, \
    false_discovery_rate, time_between_false_alarms, mean_time_ratio, missed_detection_rate
from float.data.data_loader import DataLoader
import unittest


class TestChangeDetectionEvaluator(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        self.data_loader = DataLoader(path=f'../datasets/spambase.csv', target_col=-1)
        known_drifts = [round(self.data_loader.stream.n_samples * 0.2), round(self.data_loader.stream.n_samples * 0.4),
                        round(self.data_loader.stream.n_samples * 0.6), round(self.data_loader.stream.n_samples * 0.8)]
        batch_size = 10
        self.erics = ERICS(self.data_loader.stream.n_features)
        self.change_detector_evaluator = ChangeDetectionEvaluator(measure_funcs=[detected_change_rate,
                                                                                 missed_detection_rate,
                                                                                 false_discovery_rate,
                                                                                 time_between_false_alarms,
                                                                                 detection_delay,
                                                                                 mean_time_ratio],
                                                                  known_drifts=known_drifts,
                                                                  batch_size=batch_size,
                                                                  n_total=self.data_loader.stream.n_samples,
                                                                  n_delay=list(range(100, 1000)),
                                                                  n_init_tolerance=100)

    def test_init(self):
        for measure_func in self.change_detector_evaluator.measure_funcs:
            self.assertTrue(measure_func.__name__ in self.change_detector_evaluator.result.keys())

    def test_run(self):
        X, y = self.data_loader.get_data(50)
        self.erics.partial_fit(X, y)
        self.change_detector_evaluator.run(self.erics.drifts)
        for measure_func in self.change_detector_evaluator.measure_funcs:
            self.assertTrue(len(self.change_detector_evaluator.result[measure_func.__name__]['measures']) > 0,
                            msg='run() adds a value to the measure dict')
            self.assertIsInstance(self.change_detector_evaluator.result[measure_func.__name__]['mean'], (float, int),
                                  msg='run() returns a numeric mean value')
            self.assertIsInstance(self.change_detector_evaluator.result[measure_func.__name__]['var'], (float, int),
                                  msg='run() returns a numeric variance value')
