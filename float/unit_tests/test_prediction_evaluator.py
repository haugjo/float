from float.data.data_loader import DataLoader
from float.prediction.evaluation.measures import mean_drift_restoration_time, mean_drift_performance_deterioration, noise_variability
from float.prediction.evaluation.prediction_evaluator import PredictionEvaluator
from float.prediction.skmultiflow.skmultiflow_classifier import SkmultiflowClassifier
import numpy as np
from sklearn.metrics import accuracy_score, zero_one_loss
from skmultiflow.neural_networks.perceptron import PerceptronMask
import unittest


class TestPredictionEvaluator(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        self.rng = np.random.default_rng(seed=0)
        self.data_loader = DataLoader(path=f'../data/datasets/spambase.csv', target_col=-1)
        known_drifts = [round(self.data_loader.stream.n_samples * 0.2), round(self.data_loader.stream.n_samples * 0.4),
                        round(self.data_loader.stream.n_samples * 0.6), round(self.data_loader.stream.n_samples * 0.8)]
        batch_size = 10
        self.predictor = SkmultiflowClassifier(PerceptronMask(), self.data_loader.stream.target_values, reset_after_drift=True)  # todo: can we get rid of the target values parameter?
        self.prediction_evaluator = PredictionEvaluator([accuracy_score, zero_one_loss, mean_drift_performance_deterioration, mean_drift_restoration_time, noise_variability],
                                                        decay_rate=0.1,
                                                        window_size=10,
                                                        known_drifts=known_drifts,
                                                        batch_size=batch_size,
                                                        interval=10)

    def test_init(self):
        for measure_func in self.prediction_evaluator.measure_funcs:
            self.assertTrue(measure_func.__name__ in self.prediction_evaluator.result.keys())

    def test_run(self):
        X, y = self.data_loader.get_data(50)
        self.predictor.partial_fit(X, y)
        y_pred = self.predictor.predict(X)
        self.prediction_evaluator.run(y, y_pred, X, self.predictor, self.rng)
        for measure_func in self.prediction_evaluator.measure_funcs:
            self.assertTrue(len(self.prediction_evaluator.result[measure_func.__name__]['measures']) > 0, msg='run() adds a value to the measure dict')
            self.assertTrue(len(self.prediction_evaluator.result[measure_func.__name__]['measures']) > 0, msg='run() adds a value to the mean dict')
            self.assertTrue(len(self.prediction_evaluator.result[measure_func.__name__]['measures']) > 0, msg='run() adds a value to the var dict')
