from skmultiflow.core import ClassifierMixin
from float.prediction.base_predictor import BasePredictor


class SkmultiflowClassifier(BasePredictor):
    """
    Wrapper for skmultiflow predictor classes.
    """
    def __init__(self, model, classes):
        """
        Initializes the skmultiflow PerceptronMask.

        Args:
            model (ClassifierMixin): the sklearn classifier to be used for prediction
            classes (list): the list of classes in the data
        """
        super().__init__()
        self.model = model
        self.classes = classes

    def partial_fit(self, X, y, sample_weight=None):
        self.model.partial_fit(X, y, classes=self.classes, sample_weight=sample_weight)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
