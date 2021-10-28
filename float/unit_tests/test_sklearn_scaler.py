import unittest
from float.data import DataLoader
from float.data.preprocessing import SklearnScaler
from sklearn.preprocessing import MinMaxScaler


class TestSklearnScaler(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        self.data_loader = DataLoader(path='../data/datasets/spambase.csv', target_col=-1)
        self.scaler = SklearnScaler(MinMaxScaler())
        X, y = self.data_loader.get_data(10)
        self.scaler.partial_fit(X)

    def test_init(self):
        self.assertEqual(self.scaler.reset_after_drift, False, msg='attribute reset_after_drift is initialized correctly')
        self.assertIsInstance(self.scaler.scaler_obj, MinMaxScaler, msg='attribute scaler_obj is initialized correctly')
        self.assertIsInstance(self.scaler._has_partial_fit, bool, msg='attribute has_partial_fit is initialized correctly')
        self.assertIsInstance(self.scaler._must_be_fitted, bool, msg='attribute must_be_fitted is initialized correctly')

    def test_partial_fit(self):
        data_range = self.scaler.scaler_obj.data_range_
        X, y = self.data_loader.get_data(10)
        self.scaler.partial_fit(X)
        self.assertFalse((data_range == self.scaler.scaler_obj.data_range_).all(), msg='partial_fit() updates the data_range attribute of the scaler_obj')

    def test_transform(self):
        X, y = self.data_loader.get_data(10)
        self.scaler.partial_fit(X)
        X_scaled = self.scaler.transform(X)
        self.assertEqual(X.shape, X_scaled.shape, msg='transform() preserves shape of X')
        self.assertTrue(((X_scaled >= 0) | (X_scaled <= 1)).all(), msg='transform() updates X to only include values between 0 and 1')

    def test_reset(self):
        self.scaler._must_be_fitted = False
        self.scaler.reset()
        self.assertTrue(self.scaler._must_be_fitted, msg='reset() sets attribute must_be_fitted to true')
