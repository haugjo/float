from float.pipeline.pipeline import Pipeline
from float.data.data_loader import DataLoader
from float.feature_selection import FeatureSelector
from float.concept_drift_detection import ConceptDriftDetector
from float.prediction import Predictor
from float.evaluation.evaluator import Evaluator


class HoldoutPipeline(Pipeline):
    """
    Pipeline which implements the holdout evaluation.

     Attributes:
        data_loader (DataLoader): DataLoader object
        feature_selector (FeatureSelector): FeatureSelector object
        concept_drift_detector (ConceptDriftDetector): ConceptDriftDetector object
        predictor (Predictor): Predictor object
        evaluator (Evaluator): Evaluator object
    """
    def __init__(self, data_loader=None, feature_selector=None, concept_drift_detector=None, predictor=None,
                 evaluator=None):
        """
        Initializes the pipeline and checks the input parameters.
        """
        super().__init__(data_loader, feature_selector, concept_drift_detector, predictor, evaluator)
        self.data_loader = data_loader
        self.feature_selector = feature_selector
        self.concept_drift_detector = concept_drift_detector
        self.predictor = predictor
        self.evaluator = evaluator

    def run(self):
        pass
