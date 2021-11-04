from float.change_detection.evaluation import ChangeDetectionEvaluator
from float.change_detection.skmultiflow import SkmultiflowChangeDetector
from float.data import DataLoader
from float.feature_selection import OFS
from float.feature_selection.evaluation import FeatureSelectionEvaluator
from float.feature_selection.evaluation.measures import nogueira_stability
from float.pipeline import PrequentialPipeline
from float.prediction.evaluation import PredictionEvaluator
from float.prediction.skmultiflow import SkmultiflowClassifier
import numpy as np
from sklearn.metrics import zero_one_loss
from skmultiflow.drift_detection import ADWIN
from skmultiflow.neural_networks import PerceptronMask
import unittest


class TestPrequentialPipeline(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        self.data_loader = DataLoader(path='../data/datasets/spambase.csv', target_col=-1)
        known_drifts = [round(self.data_loader.stream.n_samples * 0.2), round(self.data_loader.stream.n_samples * 0.4),
                        round(self.data_loader.stream.n_samples * 0.6), round(self.data_loader.stream.n_samples * 0.8)]
        batch_size = 10
        self.prequential_pipeline = PrequentialPipeline(data_loader=self.data_loader,
                                                        predictor=SkmultiflowClassifier(PerceptronMask(), classes=self.data_loader.stream.target_values),
                                                        prediction_evaluator=PredictionEvaluator([zero_one_loss]),
                                                        feature_selector=OFS(n_total_features=self.data_loader.stream.n_features, n_selected_features=10),
                                                        feature_selection_evaluator=FeatureSelectionEvaluator([nogueira_stability]),
                                                        change_detector=SkmultiflowChangeDetector(ADWIN()),
                                                        change_detection_evaluator=ChangeDetectionEvaluator([], batch_size=batch_size, known_drifts=known_drifts, n_total=self.data_loader.stream.n_samples, n_delay=list(range(100, 1000)), n_init_tolerance=100),
                                                        batch_size=10,
                                                        n_max=self.data_loader.stream.n_samples - 10,
                                                        random_state=0)

    def test_init(self):
        self.assertEqual(self.prequential_pipeline.start_time, 0, msg='attribute start_time is initialized correctly')
        self.assertEqual(self.prequential_pipeline.time_step, 0, msg='attribute time_step is initialized correctly')
        self.assertEqual(self.prequential_pipeline.n_total, 0, msg='attribute n_total is initialized correctly')

        with self.assertRaises(AttributeError, msg='AttributeError when passing neither Feature Selector, '
                                                   'Change Detector nor Predictor objects'):
            PrequentialPipeline(self.data_loader)

        with self.assertRaises(AttributeError, msg='AttributeError when an error-based Change Detector is passed but no Predictor'):
            PrequentialPipeline(self.data_loader, change_detector=SkmultiflowChangeDetector(ADWIN()))

        data_loader = DataLoader(path='../data/datasets/gas.csv', target_col=-1)
        with self.assertRaises(AttributeError, msg='AttributeError when a multiclass dataset is used but the Feature Selector'
                                                   'does not support multiclass targets'):
            PrequentialPipeline(data_loader=data_loader,
                                feature_selector=OFS(n_total_features=data_loader.stream.n_features, n_selected_features=10))

    def test_run(self):
        self.prequential_pipeline.run()
        self.assertIsInstance(self.prequential_pipeline.predictor.model.classifier.coef_, np.ndarray, msg='run() sets the classifier\'s weights')
        self.assertEqual(len(self.prequential_pipeline.prediction_evaluator.training_comp_times), self.prequential_pipeline.time_step, msg='run() adds a prediction training computation time for each time step')
        self.assertEqual(len(self.prequential_pipeline.prediction_evaluator.result['zero_one_loss']['measures']), self.prequential_pipeline.time_step, msg='run() adds a prediction evaluation measure for each time step')
        self.assertEqual(len(self.prequential_pipeline.feature_selector.selected_features_history), self.prequential_pipeline.time_step, msg='run() adds a list of selected features for every time step')
        self.assertEqual(len(self.prequential_pipeline.feature_selection_evaluator.comp_times), self.prequential_pipeline.time_step, msg='run() adds a feature selection computation time for each time step')
        self.assertEqual(len(self.prequential_pipeline.feature_selection_evaluator.result['nogueira_stability']['measures']), self.prequential_pipeline.time_step, msg='run() adds a feature selection evaluation measure for each time step')
        self.assertEqual(len(self.prequential_pipeline.change_detection_evaluator.comp_times), self.prequential_pipeline.time_step, msg='run() adds a change detection computation time for each time step')
        self.assertEqual(self.prequential_pipeline.data_loader.stream.sample_idx, 0, msg='run() restarts the data stream')