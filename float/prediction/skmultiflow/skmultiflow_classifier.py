from skmultiflow.core import ClassifierMixin
from float.prediction.base_predictor import BasePredictor


class SkmultiflowClassifier(BasePredictor):
    """
    Wrapper for skmultiflow predictor classes.
    """
    def __init__(self, model, classes, reset_after_drift=False):
        """
        Initializes the skmultiflow PerceptronMask.

        Args:
            model (ClassifierMixin): the sklearn classifier to be used for prediction
            classes (list): the list of classes in the data
            reset_after_drift (bool): indicates whether to reset the predictor after a drift was detected
        """
        super().__init__(reset_after_drift)
        self.model = model
        self.classes = classes

    def partial_fit(self, X, y, sample_weight=None):
        self.model.partial_fit(X, y, classes=self.classes, sample_weight=sample_weight)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def reset(self, X, y):
        """
        Reset and retrain on current sample
        Args:
            X (np.ndarray): data samples to train the model with
            y (np.ndarray): target values for all samples in X
        """
        self.model.reset()
        self.partial_fit(X, y)
