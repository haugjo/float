import float.visualization as fvis
import matplotlib.pyplot as plt
import numpy as np
import unittest


class TestConceptDriftDetectionScatter(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        self.sample_detected_drifts_list = [list(np.random.randint(0, 100, np.random.randint(0, 50, 1))) for _ in range(3)]

    def test_concept_drift_detection_scatter(self):
        ax = fvis.concept_drift_detection_scatter(detected_drifts=self.sample_detected_drifts_list,
                                                  model_names=[f'Model {i}' for i in range(3)],
                                                  n_samples=1000,
                                                  known_drifts=[200, 400, 600, 800],
                                                  batch_size=10,
                                                  n_pretrain=0)
        self.assertIsInstance(ax, plt.Axes, msg='concept_drift_detection_scatter() returns an object of type Axes')
        self.assertEqual(len(ax.collections), 3, msg='concept_drift_detection_scatter() draws the correct amount of plots')
        self.assertEqual(ax.get_xlabel(), 'Time Step $t$', msg='concept_drift_detection_scatter() sets the correct xlabel')
        self.assertEqual([text.get_text() for text in ax.get_yticklabels()], [f'Model {i}' for i in range(3)], msg='concept_drift_detection_scatter() sets the correct yticklabels')
        self.assertEqual([text.get_text() for text in ax.get_legend().texts], ['Known drifts', 'Detected drifts'], msg='concept_drift_detection_scatter() sets the right legend texts')
