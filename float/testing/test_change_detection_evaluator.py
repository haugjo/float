import unittest
from float.data.data_loader import DataLoader
from float.change_detection.erics import ERICS
from float.change_detection.evaluation.measures import time_to_detection, detected_change_rate, \
    false_discovery_rate, time_between_false_alarms, mean_time_ratio, missed_detection_rate
from float.change_detection.evaluation.change_detection_evaluator import ChangeDetectionEvaluator


class TestChangeDetectionEvaluator(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
