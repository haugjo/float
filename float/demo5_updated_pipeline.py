from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.drift_detection.eddm import EDDM
from skmultiflow.drift_detection.ddm import DDM as DDM_scikit
from skmultiflow.neural_networks.perceptron import PerceptronMask
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, zero_one_loss

from float.data import DataLoader
from float.change_detection import ERICS, SkmultiflowChangeDetector
from float.change_detection.tornado import PageHinkley, DDM
from float.change_detection.evaluation import ChangeDetectionEvaluator
from float.change_detection.evaluation.measures import time_to_detection, detected_change_rate, \
    false_discovery_rate, time_between_false_alarms, mean_time_ratio, missed_detection_rate
from float.pipeline import PrequentialPipeline
from float.prediction import SkmultiflowPerceptron
from float.visualization import Visualizer

### Initialize Data Loader ###
data_loader = DataLoader(None, f'data/datasets/spambase.csv', target_col=0)

known_drifts = [round(data_loader.stream.n_samples * 0.2), round(data_loader.stream.n_samples * 0.4),
                round(data_loader.stream.n_samples * 0.6), round(data_loader.stream.n_samples * 0.8)]
batch_size = 10
feature_names = data_loader.stream.feature_names

concept_drift_detector_names = ['ADWIN', 'EDDM', 'DDM_sk', 'DDM', 'ERICS', 'Page Hinkley']  # Todo: Remove, and use class names instead?
concept_drift_detectors = [SkmultiflowChangeDetector(ADWIN(delta=0.6)),
                           SkmultiflowChangeDetector(EDDM()),
                           SkmultiflowChangeDetector(DDM_scikit()),
                           DDM(),
                           ERICS(data_loader.stream.n_features),
                           PageHinkley()]

### Initialize Predictor ###
predictor = SkmultiflowPerceptron(PerceptronMask(),
                                  data_loader.stream.target_values,
                                  evaluation_metrics={'Accuracy': accuracy_score,
                                                      '0-1 Loss': zero_one_loss},
                                  decay_rate=0.5, window_size=5)

# Todo: allow to provide multiple change detectors at once?
cd_evaluator = dict()

for concept_drift_detector_name, concept_drift_detector in zip(concept_drift_detector_names, concept_drift_detectors):
    ### Initialize Concept Drift Detector ###
    cd_evaluator[concept_drift_detector_name] = ChangeDetectionEvaluator(measures=[detected_change_rate,
                                                                                   missed_detection_rate,
                                                                                   false_discovery_rate,
                                                                                   time_between_false_alarms,
                                                                                   time_to_detection,
                                                                                   mean_time_ratio],
                                                                         known_drifts=known_drifts,
                                                                         batch_size=batch_size,
                                                                         n_samples=data_loader.stream.n_samples,
                                                                         n_delay=list(range(100, 1000)),
                                                                         n_init_tolerance=100)

    ### Initialize and run Prequential Pipeline ###
    prequential_pipeline = PrequentialPipeline(data_loader=data_loader,
                                               feature_selector=None,
                                               concept_drift_detector=concept_drift_detector,
                                               change_detection_evaluator=cd_evaluator[concept_drift_detector_name],
                                               predictor=predictor,
                                               batch_size=batch_size,
                                               max_n_samples=data_loader.stream.n_samples,
                                               known_drifts=known_drifts)
    prequential_pipeline.run()

visualizer = Visualizer(
    [concept_drift_detector.global_drifts for concept_drift_detector in concept_drift_detectors],
    concept_drift_detector_names, 'drift_detection')
visualizer.draw_concept_drifts(data_loader.stream, known_drifts, batch_size,
                               plot_title=f'Concept Drifts For Data Set spambase, Predictor Perceptron, Feature Selector FIRES')
plt.show()

visualizer = Visualizer(
    [cd_eval.result['detected_change_rate']['measures'] for cd_eval in cd_evaluator.values()],
    concept_drift_detector_names, 'change_detection'
)
visualizer.plot(
    plot_title=f'Concept Drift True Positive Rate For Data Set spambase, Predictor Perceptron, Feature Selector FIRES')
plt.show()
