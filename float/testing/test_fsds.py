import unittest
from float.data.data_loader import DataLoader
from float.feature_selection.evaluation.measures import nogueira_stability
from float.feature_selection.fsds import FSDS


class TestFSDS(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        data_loader = DataLoader(file_path='../data/datasets/spambase.csv', target_col=0)
        evaluation_metrics = {'Nogueira Stability Measure': (nogueira_stability,
                                                             {'n_total_features': data_loader.stream.n_features, 'nogueira_window_size': 10})}
        self.fsds = FSDS(data_loader.stream.n_features, 10, evaluation_metrics)

    def test_init(self):
        pass

    def test_weight_features(self):
        pass

    def test_select_features(self):
        pass

    def test_evaluate(self):
        pass

    def test_get_nogueira_stability_measure(self):
        pass

    def test_get_reference_value(self):
        pass
