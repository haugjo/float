from float.data import DataLoader
from float.prediction.skmultiflow import SkmultiflowClassifier
import numpy as np
from skmultiflow.core import ClassifierMixin
from skmultiflow.neural_networks import PerceptronMask
import unittest


class TestSkmultiflowClassifier(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        self.rng = np.random.default_rng(seed=0)
        self.data_loader = DataLoader(path='../datasets/spambase.csv', target_col=-1)
        self.skmultiflow_classifier = SkmultiflowClassifier(model=PerceptronMask(), classes=self.data_loader.stream.target_values)

    def test_init(self):
        self.assertEqual(self.skmultiflow_classifier.reset_after_drift, False,
                         msg='attribute reset_after_drift is initialized correctly')
        self.assertIsInstance(self.skmultiflow_classifier.model, ClassifierMixin,
                              msg='attribute model is initialized correctly')
        self.assertIsInstance(self.skmultiflow_classifier.classes, list,
                              msg='attribute classes is initialized correctly')

    def test_partial_fit(self):
        X, y = self.data_loader.get_data(10)
        self.skmultiflow_classifier.partial_fit(X, y)
        self.assertTrue(hasattr(self.skmultiflow_classifier.model.classifier, 'coef_'),
                        msg='partial_fit() updates the weights of the underlying model')

    def test_predict(self):
        X, y = self.data_loader.get_data(10)
        self.skmultiflow_classifier.partial_fit(X, y)
        X, y = self.data_loader.get_data(10)
        self.assertEqual(y.shape, self.skmultiflow_classifier.predict(X).shape,
                         msg='predict() returns the correct shape')
        self.assertTrue((np.isin(self.skmultiflow_classifier.predict(X), self.skmultiflow_classifier.classes)).all(),
                        msg='predict() only returns values corresponding to the specified classes')

    def test_predict_prob(self):
        X, y = self.data_loader.get_data(10)
        self.skmultiflow_classifier.partial_fit(X, y)
        X, y = self.data_loader.get_data(10)
        self.assertEqual((y.shape[0], 2), self.skmultiflow_classifier.predict_proba(X).shape,
                         msg='predict_proba() returns the correct shape')
        proba = self.skmultiflow_classifier.predict_proba(X).flatten()
        self.assertTrue(((proba >= 0) | (proba <= 1)).all(), msg='predict_proba() only returns values between 0 and 1')

    def test_reset(self):
        X, y = self.data_loader.get_data(10)
        self.skmultiflow_classifier.partial_fit(X, y)
        self.skmultiflow_classifier.reset()
        self.assertFalse(hasattr(self.skmultiflow_classifier.model.classifier, 'coef_'),
                         msg='reset() deletes the weights of the underlying model')
        self.assertFalse(self.skmultiflow_classifier.has_been_trained, msg='classifier has not been trained')
