import unittest
from float.data.data_loader import DataLoader
from float.change_detection.erics import ERICS


class TestERICS(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        self.data_loader = DataLoader(path='../data/datasets/spambase.csv', target_col=-1)
        self.erics = ERICS(self.data_loader.stream.n_features)

    def test_init(self):
        self.assertFalse(self.erics._drift_detected, msg='attribute global_drift_detected initialized correctly')
        self.assertFalse(self.erics._partial_drift_detected, msg='attribute partial_drift_detected initialized correctly')

    def test_partial_fit(self):
        X, y = self.data_loader.get_data(10)
        mu_w, sigma_w, param_sum, global_info_ma, partial_info_ma = self.erics._mu_w, self.erics._sigma_w, self.erics._param_sum.copy(), self.erics._info_ma.copy(), self.erics._partial_info_ma.copy()
        self.erics.partial_fit(X, y)
        self.assertFalse((mu_w[-1, :] == self.erics._mu_w[-1, :]).all(), msg='partial_fit() updates the attribute mu_w')
        self.assertTrue((mu_w[1:, :] == self.erics._mu_w[:-1, :]).all(), msg='partial_fit() adds new entry and drops oldest entry from attribute mu_w')
        self.assertFalse((sigma_w[-1, :] == self.erics._sigma_w[-1, :]).all(), msg='partial_fit() updates the attribute sigma_w')
        self.assertTrue((sigma_w[1:, :] == self.erics._sigma_w[:-1, :]).all(), msg='partial_fit() adds new entry and drops oldest entry from attribute sigma_w')
        self.assertFalse((param_sum == self.erics._param_sum).all(), msg='partial_fit() updates the attribute param_sum')
        self.assertEqual(len(global_info_ma) + 1, len(self.erics._info_ma), msg='partial_fit() adds new entry to attribute global_info_ma')
        self.assertEqual(len(partial_info_ma) + 1, len(self.erics._partial_info_ma), msg='partial_fit() adds new entry to attribute partial_info_ma')

    def test_detect_change(self):
        for i in range(10):
            X, y = self.data_loader.get_data(10)
            self.erics.partial_fit(X, y)
            self.assertEqual(self.erics._drift_detected, self.erics.detect_change(), msg='detected_global_change() returns if there was a global change')

    def test_detect_partial_change(self):
        for i in range(10):
            X, y = self.data_loader.get_data(10)
            self.erics.partial_fit(X, y)
            self.assertEqual(self.erics._partial_drift_detected, self.erics.detect_partial_change()[0], msg='detected_global_change() returns if there was a partial change')
