import warnings
import traceback
from float.pipeline.pipeline import Pipeline
from float.data.data_loader import DataLoader
from float.feature_selection import FeatureSelector
from float.concept_drift_detection import ConceptDriftDetector
from float.prediction import Predictor


class PrequentialPipeline(Pipeline):
    """
    Pipeline which implements the test-then-train evaluation.
    """
    def __init__(self, data_loader=None, feature_selector=None, concept_drift_detector=None, predictor=None,
                 max_n_samples=100000, batch_size=100, n_pretrain_samples=100, known_drifts=None):
        """
        Initializes the pipeline.

        Args:
            data_loader (DataLoader): DataLoader object
            feature_selector (FeatureSelector | None): FeatureSelector object
            concept_drift_detector (ConceptDriftDetector | None): ConceptDriftDetector object
            predictor (Predictor | None): Predictor object
            max_n_samples (int): maximum number of observations used in the evaluation
            batch_size (int): size of one batch (i.e. no. of observations at one time step)
            n_pretrain_samples (int): no. of observations used for initial training of the predictive model
            known_drifts (list): list of known concept drifts for this stream
        """
        super().__init__(data_loader, feature_selector, concept_drift_detector, predictor, max_n_samples,
                         batch_size, n_pretrain_samples, known_drifts)

    def run(self):
        """
        Runs the pipeline.

        Returns:
            list[Evaluator]: the list of evaluators
        """
        if (self.data_loader.stream.n_remaining_samples() > 0) and \
                (self.data_loader.stream.n_remaining_samples() < self.max_n_samples):
            self.max_n_samples = self.data_loader.stream.n_samples
            warnings.warn('Parameter max_n_samples exceeds the size of data_loader and will be automatically reset.',
                          stacklevel=2)

        self._start_evaluation()
        self._test_then_train()
        self._finish_evaluation()

        return self.feature_selector, self.concept_drift_detector, self.predictor

    def _test_then_train(self):
        """
        Test-then-train evaluation
        """
        while self.n_global_samples < self.max_n_samples:
            try:
                self._run_single_training_iteration()
            except BaseException:
                traceback.print_exc()
                break
