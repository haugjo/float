from float.change_detection.skmultiflow import SkmultiflowChangeDetector
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
from skmultiflow.drift_detection import DDM
import unittest


class TestSkmultiflowChangeDetector(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        ddm = DDM()
        self.skmultiflow_change_detector = SkmultiflowChangeDetector(ddm)

    def test_init(self):
        self.assertIsInstance(self.skmultiflow_change_detector.detector, BaseDriftDetector,
                              msg='attribute detector is initialized correctly')

        with self.assertRaises(TypeError, msg="TypeError when passing unsupported change detector."):
            SkmultiflowChangeDetector(None)

    def test_reset(self):
        self.skmultiflow_change_detector.detector.in_concept_change = True
        self.skmultiflow_change_detector.detector.in_warning_zone = True
        self.skmultiflow_change_detector.detector.estimation = 999
        self.skmultiflow_change_detector.detector.delay = 999
        self.skmultiflow_change_detector.detector.sample_count = 999
        self.skmultiflow_change_detector.reset()
        self.assertEqual(self.skmultiflow_change_detector.detector.in_concept_change, False,
                         "reset() sets in_concept_change to False")
        self.assertEqual(self.skmultiflow_change_detector.detector.in_warning_zone, False,
                         "reset() sets in_warning_zone to False")
        self.assertEqual(self.skmultiflow_change_detector.detector.estimation, 0.0, "reset() sets estimation to 0.0")
        self.assertEqual(self.skmultiflow_change_detector.detector.delay, 0.0, "reset() sets delay to 0.0")
        self.assertEqual(self.skmultiflow_change_detector.detector.sample_count, 1, "reset() sets sample_count to 1")

    def test_detect_change(self):
        self.skmultiflow_change_detector.partial_fit([True, False, True])
        self.assertIsInstance(self.skmultiflow_change_detector.detect_change(), bool,
                              msg="detect change returns bool indicator.")

    def test_detect_partial_change(self):
        self.assertEqual(self.skmultiflow_change_detector.detect_partial_change(), (False, []),
                         "detected_partial_change() returns False initially")

    def test_detect_warning_zone(self):
        self.assertEqual(self.skmultiflow_change_detector.detect_warning_zone(), False,
                         "detected_warning_zone() returns False initially")

    def test_partial_fit(self):
        sample_count = self.skmultiflow_change_detector.detector.sample_count
        miss_prob = self.skmultiflow_change_detector.detector.miss_prob
        self.skmultiflow_change_detector.partial_fit([True, False, True])
        self.assertEqual(self.skmultiflow_change_detector.detector.sample_count, sample_count + 3,
                         "partial_fit() increases sample_count by 3")
        self.assertNotEqual(self.skmultiflow_change_detector.detector.miss_prob, miss_prob,
                            "partial_fit() updates miss_prob")
        self.assertIsInstance(self.skmultiflow_change_detector.detect_change(), bool,
                              msg="detect_change returns a bool indicator.")
        self.assertIsInstance(self.skmultiflow_change_detector.detect_warning_zone(), bool,
                              msg="detect_warning returns a bool indicator.")
        self.assertIsInstance(self.skmultiflow_change_detector.detect_partial_change(), tuple,
                              msg="detect partial change returns a tuple.")
