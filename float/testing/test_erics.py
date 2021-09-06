import unittest
from float.data.data_loader import DataLoader
from float.concept_drift_detection.concept_drift_detector import ConceptDriftDetector
from float.concept_drift_detection.erics import ERICS


class TestERICS(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        data_loader = DataLoader(file_path='../data/datasets/spambase.csv', target_col=0)
        known_drifts = [round(data_loader.stream.n_samples * 0.2), round(data_loader.stream.n_samples * 0.4)]
        batch_size = 10
        evaluation_metrics = {
            'Delay': (
                ConceptDriftDetector.get_average_delay,
                {'known_drifts': known_drifts, 'batch_size': batch_size, 'max_n_samples': data_loader.stream.n_samples}),
            'TPR': (
                ConceptDriftDetector.get_tpr,
                {'known_drifts': known_drifts, 'batch_size': batch_size, 'max_delay_range': 100}),
            'FDR': (
                ConceptDriftDetector.get_fdr,
                {'known_drifts': known_drifts, 'batch_size': batch_size, 'max_delay_range': 100}),
            'Precision': (
                ConceptDriftDetector.get_precision,
                {'known_drifts': known_drifts, 'batch_size': batch_size, 'max_delay_range': 100})
        }
        self.erics = ERICS(data_loader.stream.n_features, evaluation_metrics)

    def test_init(self):
        pass

    def test_reset(self):
        pass

    def test_partial_fit(self):
        pass

    def test_update_param_sum(self):
        pass

    def test_compute_moving_average(self):
        pass

    def test_detect_drift(self):
        pass

    def test_update_probit(self):
        pass

    def test_detected_global_change(self):
        pass

    def test_detected_partial_change(self):
        pass

    def test_detected_warning_zone(self):
        pass

    def test_get_length_estimation(self):
        pass

    def test_evaluate(self):
        pass

    def test_get_average_delay(self):
        pass

    def test_get_tpr_fdr_and_precision(self):
        pass

    def test_get_tpr(self):
        pass

    def test_get_fdr(self):
        pass

    def test_get_precision(self):
        pass
