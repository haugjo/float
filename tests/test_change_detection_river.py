from float.change_detection.river import RiverChangeDetector
from river.base import DriftDetector
from river.drift import DDM
import unittest


class TestRiverChangeDetector(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        ddm = DDM()
        self.river_change_detector = RiverChangeDetector(ddm)

    def test_init(self):
        self.assertIsInstance(self.river_change_detector.detector, DriftDetector,
                              msg='attribute detector is initialized correctly')

        with self.assertRaises(TypeError, msg="TypeError when passing unsupported change detector."):
            RiverChangeDetector(None)

    def test_reset(self):
        self.river_change_detector.detector._in_concept_change = True
        self.river_change_detector.detector.sample_count = 999
        self.river_change_detector.detector.miss_prob = 999
        self.river_change_detector.detector.miss_std = 999
        self.river_change_detector.reset()
        self.assertEqual(self.river_change_detector.detector._in_concept_change, False,
                         "reset() sets _in_concept_change to False")
        self.assertEqual(self.river_change_detector.detector.sample_count, 1,
                         "reset() sets sample_count to 1")
        self.assertEqual(self.river_change_detector.detector.miss_prob, 1.0, "reset() sets miss_prob to 1.0")
        self.assertEqual(self.river_change_detector.detector.miss_std, 0.0, "reset() sets miss_std to 0.0")

    def test_detect_change(self):
        self.river_change_detector.partial_fit([True, False, True])
        self.assertIsInstance(self.river_change_detector.detect_change(), bool,
                              msg="detect change returns bool indicator.")

    def test_detect_partial_change(self):
        self.assertEqual(self.river_change_detector.detect_partial_change(), (False, []),
                         "detected_partial_change() returns False initially")

    def test_detect_warning_zone(self):
        self.assertEqual(self.river_change_detector.detect_warning_zone(), False,
                         "detected_warning_zone() returns False initially")

    def test_partial_fit(self):
        sample_count = self.river_change_detector.detector.sample_count
        miss_prob = self.river_change_detector.detector.miss_prob
        self.river_change_detector.partial_fit([True, False, True])
        self.assertEqual(self.river_change_detector.detector.sample_count, sample_count + 3,
                         "partial_fit() increases sample_count by 3")
        self.assertNotEqual(self.river_change_detector.detector.miss_prob, miss_prob,
                            "partial_fit() updates miss_prob")
        self.assertIsInstance(self.river_change_detector.detect_change(), bool,
                              msg="detect_change returns a bool indicator.")
        self.assertIsInstance(self.river_change_detector.detect_warning_zone(), bool,
                              msg="detect_warning returns a bool indicator.")
        self.assertIsInstance(self.river_change_detector.detect_partial_change(), tuple,
                              msg="detect partial change returns a tuple.")
