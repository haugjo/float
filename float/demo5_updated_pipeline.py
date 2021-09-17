from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.drift_detection.eddm import EDDM
from skmultiflow.drift_detection.ddm import DDM
from skmultiflow.neural_networks.perceptron import PerceptronMask
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, zero_one_loss

from float.data import DataLoader
from float.change_detection import SkmultiflowDriftDetector, ERICS
from float.change_detection.tornado import PageHinkley
from float.change_detection.measures import ChangeDetectionEvaluator, delay, recall, false_discovery_rate
from float.pipeline import PrequentialPipeline
from float.prediction import SkmultiflowPerceptron
from float.visualization import Visualizer

### Initialize Data Loader ###
data_loader = DataLoader(None, f'data/datasets/spambase.csv', target_col=0)

known_drifts = [round(data_loader.stream.n_samples * 0.2), round(data_loader.stream.n_samples * 0.4),
                round(data_loader.stream.n_samples * 0.6), round(data_loader.stream.n_samples * 0.8)]
batch_size = 10
feature_names = data_loader.stream.feature_names

# evaluation_metrics (dict of str: function | dict of str: (function, dict)): {metric_name: metric_function} OR {metric_name: (metric_function, {param_name1: param_val1, ...})} a dictionary of metrics to be used

concept_drift_detector_names = ['ADWIN', 'EDDM', 'DDM', 'ERICS']
concept_drift_detectors = [SkmultiflowDriftDetector(ADWIN(delta=0.6)),
                           SkmultiflowDriftDetector(EDDM()),
                           SkmultiflowDriftDetector(DDM()),
                           ERICS(data_loader.stream.n_features),
                           PageHinkley()]

### Initialize Predictor ###
predictor = SkmultiflowPerceptron(PerceptronMask(),
                                  data_loader.stream.target_values,
                                  evaluation_metrics={'Accuracy': accuracy_score,
                                                      '0-1 Loss': zero_one_loss},
                                  decay_rate=0.5, window_size=5)

# Todo: allow to provide multiple change detectors at once?
for concept_drift_detector_name, concept_drift_detector in zip(concept_drift_detector_names, concept_drift_detectors):
    ### Initialize Concept Drift Detector ###
    cd_evaluator = ChangeDetectionEvaluator(measures=[delay, recall, false_discovery_rate],
                                            known_drifts=known_drifts,
                                            batch_size=batch_size,
                                            n_samples=data_loader.stream.n_samples,
                                            n_delay=[100, 500, 1000],
                                            n_init_tolerance=100)

    ### Initialize and run Prequential Pipeline ###
    prequential_pipeline = PrequentialPipeline(data_loader=data_loader,
                                               feature_selector=None,
                                               concept_drift_detector=concept_drift_detector,
                                               change_detection_evaluator=cd_evaluator,
                                               predictor=predictor,
                                               batch_size=batch_size,
                                               max_n_samples=data_loader.stream.n_samples,
                                               known_drifts=known_drifts)
    prequential_pipeline.run()

# Todo: update visualizer of measures
"""
visualizer = Visualizer(
    [concept_drift_detector.global_drifts for concept_drift_detector in concept_drift_detectors],
    concept_drift_detector_names, 'drift_detection')
visualizer.draw_concept_drifts(data_loader.stream, known_drifts, batch_size,
                               plot_title=f'Concept Drifts For Data Set spambase, Predictor Perceptron, Feature Selector FIRES')
plt.show()

visualizer = Visualizer(
    [concept_drift_detector.evaluation['TPR'] for concept_drift_detector in
     concept_drift_detectors],
    concept_drift_detector_names, 'change_detection'
)
visualizer.plot(
    plot_title=f'Concept Drift True Positive Rate For Data Set spambase, Predictor Perceptron, Feature Selector FIRES')
plt.show()
"""
