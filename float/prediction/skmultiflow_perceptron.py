from skmultiflow.neural_networks.perceptron import PerceptronMask
from float.prediction.predictor import Predictor


class SkmultiflowPerceptron(Predictor):
    """
    Wrapper for the skmultiflow PerceptronMask class.
    """
    def __init__(self, perceptron, classes, decay_rate=None, window_size=None):
        """
        Initializes the skmultiflow PerceptronMask.

        Args:
            perceptron (PerceptronMask): the sklearn perceptron to be used for prediction
            classes (list): the list of classes in the data
        """
        super().__init__(classes, decay_rate, window_size)
        self.perceptron = perceptron

    def fit(self, X, y, sample_weight=None):
        self.perceptron.fit(X, y, classes=self.classes, sample_weight=sample_weight)

    def partial_fit(self, X, y, sample_weight=None):
        self.perceptron.partial_fit(X, y, classes=self.classes, sample_weight=sample_weight)

    def predict(self, X):
        return self.perceptron.predict(X)

    def predict_proba(self, X):
        return self.perceptron.predict_proba(X)
