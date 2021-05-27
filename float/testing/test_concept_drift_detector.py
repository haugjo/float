import unittest
from skmultiflow.drift_detection.ddm import DDM
from float.concept_drift_detection.skmultiflow_drift_detector import SkmultiflowDriftDetector


class TestConceptDriftDetector(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        ddm = DDM()
        self.concept_drift_detector = SkmultiflowDriftDetector(ddm)

    def test_init(self):
        raise NotImplementedError

    def test_reset(self):
        raise NotImplementedError

    def test_detected_global_change(self):
        raise NotImplementedError

    def test_detected_warning_zone(self):
        raise NotImplementedError

    def test_get_length_estimation(self):
        raise NotImplementedError

    def test_partial_fit(self):
        raise NotImplementedError
