from float.data import DataLoader
from float.prediction.river import RiverClassifier
import numpy as np
from river.base import Classifier
from river.linear_model import Perceptron
import unittest


class TestRiverClassifier(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        self.rng = np.random.default_rng(seed=0)
        self.data_loader = DataLoader(path='../data/datasets/spambase.csv', target_col=-1)
        self.river_classifier = RiverClassifier(model=Perceptron(), features=self.data_loader.stream.feature_names)
        X, y = self.data_loader.get_data(10)
        self.river_classifier.partial_fit(X, y)

    def test_init(self):
        self.assertEqual(self.river_classifier.reset_after_drift, False, msg='attribute reset_after_drift is initialized correctly')
        self.assertIsInstance(self.river_classifier.model, Classifier, msg='attribute model is initialized correctly')
        self.assertIsInstance(self.river_classifier.features, list, msg='attribute classes is initialized correctly')

    def test_partial_fit(self):
        n_iterations = self.river_classifier.model.optimizer.n_iterations
        X, y = self.data_loader.get_data(10)
        self.river_classifier.partial_fit(X, y)
        self.assertEqual(n_iterations + 1, self.river_classifier.model.optimizer.n_iterations, msg='partial_fit() updates the optimizer of the underlying model')

    def test_predict(self):
        X, y = self.data_loader.get_data(10)
        self.assertEqual(y.shape, self.river_classifier.predict(X).shape, msg='predict() returns the correct shape')
        self.assertTrue((np.isin(self.river_classifier.predict(X), self.data_loader.stream.target_values)).all(), msg='predict() only returns values corresponding to the specified classes')

    def test_predict_prob(self):
        X, y = self.data_loader.get_data(10)
        self.assertEqual((y.shape[0], 2), self.river_classifier.predict_proba(X).shape, msg='predict_proba() returns the correct shape')
        proba = self.river_classifier.predict_proba(X)
        self.assertTrue(((proba >= 0).all() | (proba <= 1).all()).all(), msg='predict_proba() only returns values between 0 and 1')

    def test_reset(self):
        X, y = self.data_loader.get_data(10)
        self.river_classifier.reset(X, y)
        self.assertEqual(1, self.river_classifier.model.optimizer.n_iterations, msg='partial_fit() updates the optimizer of the underlying model')
