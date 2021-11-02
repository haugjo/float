import float.visualization as fvis
import matplotlib.pyplot as plt
import numpy as np
import unittest


class TestVisualization(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        self.sample_measures = [list(np.random.rand(200)) for _ in range(3)]
        self.sample_selected_features = [list(np.random.randint(0, 50, 20)) for _ in range(100)]
        self.sample_selected_features_list = [[list(np.random.randint(0, 50, 20)) for _ in range(100)] for _ in range(3)]
        self.sample_detected_drifts_list = [list(np.random.randint(0, 100, np.random.randint(0, 50, 1))) for _ in range(3)]

    def test_plot(self):
        ax = fvis.plot(measures=self.sample_measures, legend_labels=[f'Model {i}' for i in range(3)], y_label='Measure')
        self.assertIsInstance(ax, plt.Axes, msg='plot() returns an object of type Axes')
        self.assertEqual(len(ax.lines), len(self.sample_measures), msg='plot() draws the correct amount of plots')
        self.assertEqual(ax.get_xlabel(), 'Time Step $t$', msg='plot() sets the correct xlabel')
        self.assertEqual(ax.get_ylabel(), 'Measure', msg='plot() set the correct ylabel')
        self.assertEqual([text.get_text() for text in ax.get_legend().texts], [f'Model {i}' for i in range(3)], msg='plot() sets the right legend texts')

    def test_scatter(self):
        ax = fvis.plot(measures=self.sample_measures, legend_labels=[f'Model {i}' for i in range(3)], y_label='Measure')
        self.assertIsInstance(ax, plt.Axes, msg='scatter() returns an object of type Axes')
        self.assertEqual(len(ax.lines), len(self.sample_measures), msg='scatter() draws the correct amount of plots')
        self.assertEqual(ax.get_xlabel(), 'Time Step $t$', msg='scatter() sets the correct xlabel')
        self.assertEqual(ax.get_ylabel(), 'Measure', msg='scatter() set the correct ylabel')
        self.assertEqual([text.get_text() for text in ax.get_legend().texts], [f'Model {i}' for i in range(3)], msg='scatter() sets the right legend texts')

    def test_bar(self):
        ax = fvis.plot(measures=self.sample_measures, legend_labels=[f'Model {i}' for i in range(3)], y_label='Measure')
        self.assertIsInstance(ax, plt.Axes, msg='bar() returns an object of type Axes')
        self.assertEqual(len(ax.lines), len(self.sample_measures), msg='bar() draws the correct amount of plots')
        self.assertEqual(ax.get_xlabel(), 'Time Step $t$', msg='bar() sets the correct xlabel')
        self.assertEqual(ax.get_ylabel(), 'Measure', msg='bar() set the correct ylabel')
        self.assertEqual([text.get_text() for text in ax.get_legend().texts], [f'Model {i}' for i in range(3)], msg='bar() sets the right legend texts')

    def test_feature_selection_scatter(self):
        ax = fvis.feature_selection_scatter(selected_features=self.sample_selected_features)
        self.assertIsInstance(ax, plt.Axes, msg='feature_selection_scatter() returns an object of type Axes')
        self.assertEqual(ax.get_xlabel(), 'Time Step $t$', msg='feature_selection_scatter() sets the correct xlabel')
        self.assertEqual(ax.get_ylabel(), 'Feature Index', msg='feature_selection_scatter() set the correct ylabel')
        self.assertEqual(ax.get_legend().texts[0].get_text(), 'Selected Feature Indicator', msg='feature_selection_scatter() sets the right legend text')

    def test_feature_selection_bar(self):
        ax = fvis.feature_selection_bar(selected_features=self.sample_selected_features_list,
                                        model_names=[f'Model {i}' for i in range(3)],
                                        feature_names=[f'Feature {i}' for i in range(50)],
                                        top_n_features=10)
        self.assertIsInstance(ax, plt.Axes, msg='feature_selection_bar() returns an object of type Axes')
        self.assertEqual(len(ax.patches), len(self.sample_selected_features_list) * 10, msg='feature_selection_bar() draws the correct amount of bars')
        self.assertEqual(ax.get_xlabel(), 'Input Feature', msg='feature_selection_bar() sets the correct xlabel')
        self.assertEqual(ax.get_ylabel(), 'No. Times Selected', msg='feature_selection_bar() set the correct ylabel')
        self.assertEqual([text.get_text() for text in ax.get_legend().texts], [f'Model {i}' for i in range(3)], msg='feature_selection_bar() sets the right legend texts')

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
        self.assertEqual([text.get_text() for text in ax.get_yticklabels()], [f'Model {i}' for i in range(3)], msg='concept_drift_detection_scatter() set the correct yticklabels')
        self.assertEqual([text.get_text() for text in ax.get_legend().texts], ['Known drifts', 'Detected drifts'], msg='concept_drift_detection_scatter() sets the right legend texts')
