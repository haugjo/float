import unittest
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
from skmultiflow.drift_detection.ddm import DDM
from float.change_detection.skmultiflow.skmultiflow_change_detector import SkmultiflowChangeDetector


class TestSkmultiflowDriftDetector(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        ddm = DDM()
        self.skmultiflow_drift_detector = SkmultiflowChangeDetector(ddm)

    def test_init(self):
        self.assertIsInstance(self.skmultiflow_drift_detector.detector, BaseDriftDetector,
                              msg='attribute detector is intialized correctly')

    def test_reset(self):
        self.skmultiflow_drift_detector.detector.in_concept_change = True
        self.skmultiflow_drift_detector.detector.in_warning_zone = True
        self.skmultiflow_drift_detector.detector.estimation = 999
        self.skmultiflow_drift_detector.detector.delay = 999
        self.skmultiflow_drift_detector.detector.sample_count = 999
        self.skmultiflow_drift_detector.reset()
        self.assertEqual(self.skmultiflow_drift_detector.detector.in_concept_change, False,
                         "reset() sets in_concept_change to False")
        self.assertEqual(self.skmultiflow_drift_detector.detector.in_warning_zone, False,
                         "reset() sets in_warning_zone to False")
        self.assertEqual(self.skmultiflow_drift_detector.detector.estimation, 0.0, "reset() sets estimation to 0.0")
        self.assertEqual(self.skmultiflow_drift_detector.detector.delay, 0.0, "reset() sets delay to 0.0")
        self.assertEqual(self.skmultiflow_drift_detector.detector.sample_count, 1, "reset() sets sample_count to 1")

    def test_detect_change(self):
        self.assertEqual(self.skmultiflow_drift_detector.detect_change(), False,
                         "detected_global_change() returns False initially")
        self.skmultiflow_drift_detector.partial_fit(0)
        self.assertEqual(self.skmultiflow_drift_detector.detect_change(), False,
                         "detected_global_change() returns False for the same concept")
        for i in range(50):
            self.skmultiflow_drift_detector.partial_fit(0)
        self.skmultiflow_drift_detector.partial_fit(1)
        self.assertEqual(self.skmultiflow_drift_detector.detect_change(), True,
                         "detected_global_change() returns True for a different concept")

    def test_detect_partial_change(self):
        self.assertEqual(self.skmultiflow_drift_detector.detect_partial_change(), (False, []),
                         "detected_partial_change() returns False initially")

    def test_detect_warning_zone(self):
        self.assertEqual(self.skmultiflow_drift_detector.detect_warning_zone(), False,
                         "detected_warning_zone() returns False initially")

    def test_partial_fit(self):
        sample_count = self.skmultiflow_drift_detector.detector.sample_count
        miss_prob = self.skmultiflow_drift_detector.detector.miss_prob
        self.skmultiflow_drift_detector.partial_fit(0)
        self.assertEqual(self.skmultiflow_drift_detector.detector.sample_count, sample_count + 1,
                         "partial_fit() increases sample_count by 1")
        self.assertNotEqual(self.skmultiflow_drift_detector.detector.miss_prob, miss_prob,
                            "partial_fit() updates miss_prob")
