from float.data import DataLoader
from float.prediction import DynamicModelTreeClassifier
import numpy as np
import unittest


class TestDynamicModelTreeClassifier(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        self.rng = np.random.default_rng(seed=0)
        self.data_loader = DataLoader(path='../datasets/spambase.csv', target_col=-1)
        self.dmt_classifier = DynamicModelTreeClassifier(classes=self.data_loader.stream.target_values)

    def test_init(self):
        self.assertEqual(self.dmt_classifier.reset_after_drift, False,
                         msg='attribute reset_after_drift is initialized correctly')
        self.assertIsInstance(self.dmt_classifier.classes, list,
                              msg='attribute classes is initialized correctly')

    def test_partial_fit(self):
        X, y = self.data_loader.get_data(10)
        self.dmt_classifier.partial_fit(X, y)
        self.assertTrue(hasattr(self.dmt_classifier.root.linear_model, 'coef_'),
                        msg='partial_fit() updates the weights of the underlying model')

    def test_predict(self):
        X, y = self.data_loader.get_data(10)
        self.dmt_classifier.partial_fit(X, y)
        X, y = self.data_loader.get_data(10)
        self.assertEqual(y.shape, self.dmt_classifier.predict(X).shape,
                         msg='predict() returns the correct shape')
        self.assertTrue((np.isin(self.dmt_classifier.predict(X), self.dmt_classifier.classes)).all(),
                        msg='predict() only returns values corresponding to the specified classes')

    def test_predict_prob(self):
        X, y = self.data_loader.get_data(10)
        self.dmt_classifier.partial_fit(X, y)
        X, y = self.data_loader.get_data(10)
        self.assertEqual((y.shape[0], 2), self.dmt_classifier.predict_proba(X).shape,
                         msg='predict_proba() returns the correct shape')
        proba = self.dmt_classifier.predict_proba(X).flatten()
        self.assertTrue(((proba >= 0) | (proba <= 1)).all(), msg='predict_proba() only returns values between 0 and 1')

    def test_reset(self):
        X, y = self.data_loader.get_data(10)
        self.dmt_classifier.partial_fit(X, y)
        self.dmt_classifier.reset()
        self.assertIsNone(self.dmt_classifier.root, msg='reset() deletes the tree by resetting the root.')
        self.assertFalse(self.dmt_classifier.has_been_trained, msg='classifier has not been trained')
