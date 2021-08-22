from float.pipeline.pipeline import Pipeline
from float.data.data_loader import DataLoader
from float.feature_selection import FeatureSelector
from float.concept_drift_detection import ConceptDriftDetector
from float.prediction import Predictor


class HoldoutPipeline(Pipeline):
    """
    Pipeline which implements the holdout evaluation.
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
        raise NotImplementedError
