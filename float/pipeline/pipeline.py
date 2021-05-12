from abc import ABCMeta, abstractmethod
from float.data.data_loader import DataLoader
from float.feature_selection import FeatureSelector
from float.concept_drift_detection import ConceptDriftDetector
from float.prediction import Predictor
from float.evaluation.evaluator import Evaluator


class Pipeline(metaclass=ABCMeta):
    """
    Abstract base class which triggers events for different kinds of training procedures.
    """
    def __init__(self, data_loader, feature_selector, concept_drift_detector, predictor, evaluator):
        """
        Checks if the provided parameter values are sufficient to run a pipeline.
        TODO: determine which parameters are actually crucial instead of raising an error for all of them

        Raises:
            AttributeError: if a crucial parameter is missing
        """
        if type(data_loader) is not DataLoader:
            raise AttributeError('No valid DataLoader object was provided.')
        elif type(feature_selector) is not FeatureSelector:
            raise AttributeError('No valid FeatureSelector object was provided.')
        elif type(concept_drift_detector) is not ConceptDriftDetector:
            raise AttributeError('No valid ConceptDriftDetector object was provided.')
        elif type(predictor) is not Predictor:
            raise AttributeError('No valid Predictor object was provided.')
        elif type(evaluator) is not Evaluator:
            raise AttributeError('No valid Evaluator object was provided.')

    @abstractmethod
    def run(self):
        raise NotImplementedError
