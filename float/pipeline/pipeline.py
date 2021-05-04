from float.data.data_loader import DataLoader
from float.feature_selection import FeatureSelector
from float.concept_drift_detection import ConceptDriftDetector
from float.prediction import Predictor
from float.evaluation.evaluator import Evaluator


class Pipeline:
    """
    Base class which triggers events for different kinds of training procedures.

    Attributes:
        data_loaders (list[DataLoader]): list of DataLoader objects
        feature_selectors (list[FeatureSelector]): list of FeatureSelector objects
        concept_drift_detectors (list[ConceptDriftDetector]): list of ConceptDriftDetector objects
        predictors (list[Predictor]): list of Predictor objects
        evaluators (list[Evaluator]): list of Evaluator objects
    """
    def __init__(self, data_loaders, feature_selectors, concept_drift_detectors, predictors, evaluators):
        """
        Takes objects from the DataLoader, FeatureSelector, ConceptDriftDetector, Predictor and Evaluator classes.

        Args:
            data_loaders (list[DataLoader]): list of DataLoader objects
            feature_selectors (list[FeatureSelector]): list of FeatureSelector objects
            concept_drift_detectors (list[ConceptDriftDetector]): list of ConceptDriftDetector objects
            predictors (list[Predictor]): list of Predictor objects
            evaluators (list[Evaluator]): list of Evaluator objects
        """
        self.data_loaders = data_loaders
        self.feature_selectors = feature_selectors
        self.concept_drift_detectors = concept_drift_detectors
        self.predictors = predictors
        self.evaluators = evaluators
