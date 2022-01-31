from float.change_detection.evaluation import ChangeDetectionEvaluator
from float.change_detection.evaluation.measures import detection_delay
from float.change_detection.skmultiflow import SkmultiflowChangeDetector
from float.data import DataLoader
from float.feature_selection import OFS
from float.feature_selection.evaluation import FeatureSelectionEvaluator
from float.feature_selection.evaluation.measures import nogueira_stability
from float.pipeline import DistributedFoldPipeline
from float.prediction.evaluation import PredictionEvaluator
from float.prediction.skmultiflow import SkmultiflowClassifier
import copy
import numpy as np
from sklearn.metrics import zero_one_loss
from skmultiflow.drift_detection import ADWIN
from skmultiflow.neural_networks import PerceptronMask
import unittest


class TestDistributedFoldPipeline(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        self.data_loader = DataLoader(path='../data/datasets/spambase.csv', target_col=-1)
        self.known_drifts = [round(self.data_loader.stream.n_samples * 0.2),
                             round(self.data_loader.stream.n_samples * 0.4),
                             round(self.data_loader.stream.n_samples * 0.6),
                             round(self.data_loader.stream.n_samples * 0.8)]
        self.prediction_evaluator = PredictionEvaluator([zero_one_loss])
        self.feature_selection_evaluator = FeatureSelectionEvaluator([nogueira_stability])
        self.change_detection_evaluator = ChangeDetectionEvaluator([detection_delay],
                                                                   batch_size=10,
                                                                   known_drifts=self.known_drifts,
                                                                   n_total=self.data_loader.stream.n_samples,
                                                                   n_delay=list(range(100, 1000)),
                                                                   n_init_tolerance=100)

    def test_init(self):
        predictors = [
            SkmultiflowClassifier(PerceptronMask(),
                                  classes=self.data_loader.stream.target_values),
            [SkmultiflowClassifier(PerceptronMask(),
                                   classes=self.data_loader.stream.target_values),
             SkmultiflowClassifier(PerceptronMask(),
                                   classes=self.data_loader.stream.target_values),
             SkmultiflowClassifier(PerceptronMask(),
                                   classes=self.data_loader.stream.target_values),
             ]
        ]

        for pred in predictors:
            fold_pipeline = DistributedFoldPipeline(data_loader=copy.deepcopy(self.data_loader),
                                                    predictor=pred,
                                                    prediction_evaluator=copy.deepcopy(self.prediction_evaluator),
                                                    feature_selector=OFS(
                                                        n_total_features=self.data_loader.stream.n_features,
                                                        n_selected_features=10),
                                                    feature_selection_evaluator=copy.deepcopy(
                                                        self.feature_selection_evaluator),
                                                    change_detector=SkmultiflowChangeDetector(ADWIN()),
                                                    change_detection_evaluator=copy.deepcopy(
                                                        self.change_detection_evaluator),
                                                    batch_size=1,
                                                    n_max=self.data_loader.stream.n_samples - 10,
                                                    label_delay_range=(10, 20),
                                                    n_parallel_instances=10,
                                                    random_state=0)

            self.assertEqual(fold_pipeline.start_time, 0, msg='attribute start_time is initialized correctly')
            self.assertEqual(fold_pipeline.time_step, 0, msg='attribute time_step is initialized correctly')
            self.assertEqual(fold_pipeline.n_total, 0, msg='attribute n_total is initialized correctly')

            if isinstance(pred, list):
                self.assertEqual(len(fold_pipeline.predictors), len(pred) * 10,
                                 msg='n instances of the predictors have been correctly initialized.')
                self.assertEqual(len(fold_pipeline.prediction_evaluators), len(pred) * 10,
                                 msg='n instances of the prediction evaluators have been correctly initialized.')
            else:
                self.assertEqual(len(fold_pipeline.predictors), 10,
                                 msg='n predictors have been correctly initialized.')
                self.assertEqual(len(fold_pipeline.prediction_evaluators), 10,
                                 msg='n instances of the prediction evaluators have been correctly initialized.')

        with self.assertRaises(AttributeError,
                               msg='AttributeError when an unknown string is passed for validation_mode.'):
            DistributedFoldPipeline(data_loader=copy.deepcopy(self.data_loader),
                                    predictor=SkmultiflowClassifier(PerceptronMask(),
                                                                    classes=self.data_loader.stream.target_values),
                                    prediction_evaluator=copy.deepcopy(self.prediction_evaluator),
                                    validation_mode='test')

        with self.assertRaises(AttributeError, msg="AttributeError when passing no valid DataLoader object."):
            DistributedFoldPipeline(data_loader=None,
                                    predictor=SkmultiflowClassifier(PerceptronMask(),
                                                                    classes=self.data_loader.stream.target_values),
                                    prediction_evaluator=copy.deepcopy(self.prediction_evaluator))

        with self.assertRaises(AttributeError, msg="AttributeError when passing an invalid predictor object."):
            DistributedFoldPipeline(self.data_loader,
                                    predictor=[SkmultiflowClassifier(PerceptronMask(),
                                                                     classes=self.data_loader.stream.target_values),
                                               PerceptronMask()],
                                    prediction_evaluator=copy.deepcopy(self.prediction_evaluator))

        with self.assertRaises(AttributeError, msg='AttributeError when failing to provide a PredictorEvaluator object along the Predictor.'):
            DistributedFoldPipeline(self.data_loader,
                                    predictor=SkmultiflowClassifier(PerceptronMask(),
                                                                    classes=self.data_loader.stream.target_values),
                                    prediction_evaluator=None)

        with self.assertRaises(AttributeError, msg='Provided ChangeDetector is invalid.'):
            DistributedFoldPipeline(self.data_loader,
                                    predictor=SkmultiflowClassifier(PerceptronMask(),
                                                                    classes=self.data_loader.stream.target_values),
                                    prediction_evaluator=copy.deepcopy(self.prediction_evaluator),
                                    change_detector=1,
                                    change_detection_evaluator=copy.deepcopy(self.change_detection_evaluator))

        with self.assertRaises(AttributeError, msg='AttributeError when failing to provide a ChangeDetectionEvaluator along the ChangeDetector.'):
            DistributedFoldPipeline(self.data_loader,
                                    predictor=SkmultiflowClassifier(PerceptronMask(),
                                                                    classes=self.data_loader.stream.target_values),
                                    prediction_evaluator=copy.deepcopy(self.prediction_evaluator),
                                    change_detector=SkmultiflowChangeDetector(ADWIN()))

        with self.assertRaises(AttributeError, msg='AttributeError when the FeatureSelection object is invalid.'):
            DistributedFoldPipeline(self.data_loader,
                                    predictor=SkmultiflowClassifier(PerceptronMask(),
                                                                    classes=self.data_loader.stream.target_values),
                                    prediction_evaluator=copy.deepcopy(self.prediction_evaluator),
                                    feature_selector=12)

        with self.assertRaises(AttributeError, msg='AttributeError when failing to provide a FeatureSelectionEvaluator along the FeatureSelection object.'):
            DistributedFoldPipeline(self.data_loader,
                                    predictor=SkmultiflowClassifier(PerceptronMask(),
                                                                    classes=self.data_loader.stream.target_values),
                                    prediction_evaluator=copy.deepcopy(self.prediction_evaluator),
                                    feature_selector=OFS(
                                        n_total_features=self.data_loader.stream.n_features,
                                        n_selected_features=10))

        with self.assertRaises(AttributeError, msg='AttributeError when a multiclass dataset is used but the Feature Selector'
                                                   'does not support multiclass targets'):
            data_loader = DataLoader(path='../data/datasets/gas.csv', target_col=-1)
            DistributedFoldPipeline(data_loader=data_loader,
                                    predictor=SkmultiflowClassifier(PerceptronMask(),
                                                                    classes=self.data_loader.stream.target_values),
                                    prediction_evaluator=copy.deepcopy(self.prediction_evaluator),
                                    feature_selector=OFS(n_total_features=data_loader.stream.n_features, n_selected_features=10))

    def test_run(self):
        for batch_size in range(1, 100, 10):
            cd_evaluator = copy.deepcopy(self.change_detection_evaluator)
            cd_evaluator.batch_size = batch_size
            pipeline = DistributedFoldPipeline(data_loader=copy.deepcopy(self.data_loader),
                                               predictor=SkmultiflowClassifier(PerceptronMask(),
                                                                               classes=self.data_loader.stream.target_values),
                                               prediction_evaluator=copy.deepcopy(self.prediction_evaluator),
                                               feature_selector=OFS(
                                                   n_total_features=self.data_loader.stream.n_features,
                                                   n_selected_features=10),
                                               feature_selection_evaluator=copy.deepcopy(
                                                   self.feature_selection_evaluator),
                                               change_detector=SkmultiflowChangeDetector(ADWIN()),
                                               change_detection_evaluator=copy.deepcopy(
                                                   self.change_detection_evaluator),
                                               batch_size=1,
                                               n_max=self.data_loader.stream.n_samples - 10,
                                               n_parallel_instances=3,
                                               random_state=0)

            pipeline.run()
            self.assertEqual(len(pipeline.prediction_evaluators[0]), 3,
                             msg='run() maintains three parallel prediction evaluator objects.')
            self.assertIsInstance(pipeline.predictors[0][0].model.classifier.coef_, np.ndarray,
                                  msg='run() sets the classifier\'s weights')
            self.assertEqual(len(pipeline.prediction_evaluators[0][0].training_comp_times), pipeline.time_step,
                             msg='run() adds a prediction training computation time for each time step')
            self.assertEqual(len(pipeline.prediction_evaluators[0][0].result['zero_one_loss']['measures']),
                             pipeline.time_step,
                             msg='run() adds a prediction evaluation measure for each time step')
            self.assertEqual(len(pipeline.feature_selector.selected_features_history), pipeline.time_step,
                             msg='run() adds a list of selected features for every time step')
            self.assertEqual(len(pipeline.feature_selection_evaluator.comp_times), pipeline.time_step,
                             msg='run() adds a feature selection computation time for each time step')
            self.assertEqual(len(pipeline.feature_selection_evaluator.result['nogueira_stability']['measures']),
                             pipeline.time_step,
                             msg='run() adds a feature selection evaluation measure for each time step')
            self.assertEqual(len(pipeline.change_detection_evaluator.comp_times), pipeline.time_step,
                             msg='run() adds a change detection computation time for each time step')
            self.assertEqual(len(pipeline.change_detection_evaluator.result['detection_delay']['measures']),
                             len(self.change_detection_evaluator.n_delay),
                             msg='run() adds a delay measure.')
