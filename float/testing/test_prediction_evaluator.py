import unittest
from float.data.data_loader import DataLoader
from float.prediction.skmultiflow.skmultiflow_classifier import SkmultiflowClassifier
from float.prediction.evaluation.prediction_evaluator import PredictionEvaluator


class TestPredictionEvaluator(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
