from float.change_detection.tornado import Adwin, Cusum, DDM, EDDM, EWMA, FHDDM, FHDDMS, \
    FHDDMSAdd, HDDMA, HDDMW, MDDMA, MDDME, MDDMG, PageHinkley, RDDM, SeqDrift2
from float.change_detection import BaseChangeDetector
import unittest


class TestRiverChangeDetector(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        self.tornado_models = [Adwin(), Cusum(), DDM(), EDDM(), EWMA(), FHDDM(), FHDDMS(), FHDDMSAdd(), HDDMA(),
                               HDDMW(), MDDMA(), MDDME(), MDDMG(), PageHinkley(), RDDM(), SeqDrift2()]

    def test_init(self):
        for mdl in self.tornado_models:
            self.assertIsInstance(mdl, BaseChangeDetector, msg='attribute detector is initialized correctly')

    def test_detect_change(self):
        for mdl in self.tornado_models:
            mdl.partial_fit([True, False, True])
            self.assertIsInstance(mdl.detect_change(), bool, msg="detect change returns bool indicator.")

            mdl._active_change = False
            self.assertEqual(mdl.detect_change(), False,
                             msg="detect change returns False when there is no active concept drift")
            mdl._active_change = True
            self.assertEqual(mdl.detect_change(), True,
                             msg="detect change returns True when there is active concept drift")

    def test_detect_partial_change(self):
        for mdl in self.tornado_models:
            mdl.partial_fit([True, False, True])
            self.assertEqual(mdl.detect_partial_change(), (False, []),
                             msg="detect partial change returns False, as the model is not able to detect partial drift.")

    def test_detect_warning_zone(self):
        for mdl in self.tornado_models:
            if hasattr(mdl, '_active_warning'):
                mdl._active_warning = False
                self.assertEqual(mdl.detect_warning_zone(), False, msg="detect warning zone returns False")
                mdl._active_warning = True
                self.assertEqual(mdl.detect_warning_zone(), True, msg="detect warning zone returns True")
            else:
                self.assertEqual(mdl.detect_warning_zone(), False,
                                 msg="detect warning zone returns False as the model is not able to detect warnings.")

    def test_partial_fit(self):
        for mdl in self.tornado_models:
            mdl.partial_fit([True, False, True])
            self.assertIsInstance(mdl.detect_change(), bool, msg="detect_change returns a bool indicator.")
            self.assertIsInstance(mdl.detect_warning_zone(), bool, msg="detect_warning returns a bool indicator.")
            self.assertIsInstance(mdl.detect_partial_change(), tuple, msg="detect partial change returns a tuple.")
