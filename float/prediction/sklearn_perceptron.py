from sklearn.linear_model import Perceptron
from float.prediction.predictor import Predictor


class SklearnPerceptron(Predictor):
    """
    Wrapper for the sklearn Perceptron class.
    """
    def __init__(self, perceptron, classes):
        """
        Initializes the sklearn Perceptron.

        Args:
            perceptron (Perceptron): the sklearn perceptron to be used for prediction
            classes (list): the list of classes in the data
        """
        super().__init__(classes)
        self.perceptron = perceptron

    def partial_fit(self, X, y, sample_weight=None):
        self.perceptron.partial_fit(X, y, sample_weight=sample_weight, classes=self.classes)

    def predict(self, X):
        return self.perceptron.predict(X)

    def predict_proba(self, X):
        return None
