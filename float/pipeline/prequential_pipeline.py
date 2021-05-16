import warnings
from float.pipeline.pipeline import Pipeline
from float.data.data_loader import DataLoader
from float.feature_selection import FeatureSelector
from float.concept_drift_detection import ConceptDriftDetector
from float.prediction import Predictor
from float.evaluation.evaluator import Evaluator


class PrequentialPipeline(Pipeline):
    """
    Pipeline which implements the test-then-train evaluation.
    """
    def __init__(self, data_loader=None, feature_selector=None, concept_drift_detector=None, predictor=None,
                 evaluator=None, max_samples=100000, batch_size=100, pretrain_size=100, pred_metrics=None,
                 fs_metrics=None, streaming_features=None):
        """
        Initializes the pipeline.

        Args:
            data_loader (DataLoader): DataLoader object
            feature_selector (FeatureSelector): FeatureSelector object
            concept_drift_detector (ConceptDriftDetector): ConceptDriftDetector object
            predictor (Predictor): Predictor object
            evaluator (Evaluator): Evaluator object
            max_samples (int): maximum number of observations used in the evaluation
            batch_size (int): size of one batch (i.e. no. of observations at one time step)
            pretrain_size (int): no. of observations used for initial training of the predictive model
            pred_metrics (list): predictive metrics/measures
            fs_metrics (list): feature selection metrics/measures
            streaming_features (dict): (time, feature index) tuples to simulate streaming features
        """
        super().__init__(data_loader, feature_selector, concept_drift_detector, predictor, evaluator, max_samples,
                         batch_size, pretrain_size, pred_metrics, fs_metrics, streaming_features)

    def run(self):
        """
        Runs the pipeline.
        """
        if (self.data_loader.stream.n_remaining_samples() > 0) and \
                (self.data_loader.stream.n_remaining_samples() < self.max_samples):
            self.max_samples = self.data_loader.stream.n_samples
            warnings.warn('Parameter max_samples exceeds the size of data_loader and will be automatically reset.',
                          stacklevel=2)

        self._start_evaluation()
        self._test_then_train()
        self._finish_evaluation()

    def _test_then_train(self):
        """
        Test-then-train evaluation
        """
        while self.global_sample_count < self.max_samples:
            try:
                self._one_training_iteration()
            except BaseException as e:
                print(e)
                break
