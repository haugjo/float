from float.pipeline.pipeline import Pipeline
from float.data.data_loader import DataLoader
from float.feature_selection import FeatureSelector
from float.change_detection import BaseChangeDetector
from float.prediction import Predictor
import warnings
import traceback


class HoldoutPipeline(Pipeline):
    """
    Pipeline which implements the holdout evaluation.
    """
    def __init__(self, data_loader, test_set, evaluation_interval, feature_selector=None, concept_drift_detector=None, predictor=None,
                 max_n_samples=100000, batch_size=100, n_pretrain_samples=100, known_drifts=None, run=False):
        """
        Initializes the pipeline.

        Args:
            data_loader (DataLoader): DataLoader object
            test_set (np.ndarray, np.ndarray): the test samples and their labels to be used for the holdout evaluation
            evaluation_interval (int): the interval at which the predictor should be evaluated using the test set
            feature_selector (FeatureSelector | None): FeatureSelector object
            concept_drift_detector (ConceptDriftDetector | None): ConceptDriftDetector object
            predictor (Predictor | None): Predictor object
            max_n_samples (int): maximum number of observations used in the evaluation
            batch_size (int): size of one batch (i.e. no. of observations at one time step)
            n_pretrain_samples (int): no. of observations used for initial training of the predictive model
            known_drifts (list): list of known concept drifts for this stream
            run (bool): True if the run method should be executed on initialization, False otherwise
        """
        self.test_set = test_set

        super().__init__(data_loader, feature_selector, concept_drift_detector, predictor, max_n_samples,
                         batch_size, n_pretrain_samples, known_drifts, run, evaluation_interval)

    def run(self):
        """
        Runs the pipeline.
        """
        if (self.data_loader.stream.n_remaining_samples() > 0) and \
                (self.data_loader.stream.n_remaining_samples() < self.max_n_samples):
            self.max_n_samples = self.data_loader.stream.n_samples
            warnings.warn('Parameter max_n_samples exceeds the size of data_loader and will be automatically reset.',
                          stacklevel=2)

        self._start_evaluation()
        self.__holdout()
        self._finish_evaluation()

    def __holdout(self):
        """
        Holdout evaluation.
        """
        while self.n_global_samples < self.max_n_samples:
            last_iteration = False

            n_samples = self._get_n_samples()

            if self.n_global_samples + n_samples >= self.max_n_samples:
                last_iteration = True

            train_set = self.data_loader.get_data(n_samples)
            try:
                self._run_single_training_iteration(train_set, self.test_set, last_iteration)
            except BaseException:
                traceback.print_exc()
                break

            self._finish_iteration(n_samples)
