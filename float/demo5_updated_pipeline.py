from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.drift_detection.eddm import EDDM
from skmultiflow.drift_detection.ddm import DDM as DDM_scikit
from skmultiflow.neural_networks.perceptron import PerceptronMask
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, zero_one_loss
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer

from float.data import DataLoader
from float.data.scaling import SklearnScaler
from float.change_detection import ERICS, SkmultiflowChangeDetector
from float.change_detection.tornado import PageHinkley, DDM
from float.change_detection.evaluation import ChangeDetectionEvaluator
from float.change_detection.evaluation.measures import time_to_detection, detected_change_rate, \
    false_discovery_rate, time_between_false_alarms, mean_time_ratio, missed_detection_rate
from float.feature_selection.evaluation import FeatureSelectionEvaluator
from float.feature_selection.evaluation.measures import nogueira_stability
from float.feature_selection import FIRES
from float.pipeline import PrequentialPipeline
from float.prediction import SkmultiflowClassifier
from float.prediction.evaluation import PredictionEvaluator
from float.prediction.evaluation.measures import noise_variability, mean_drift_performance_decay, mean_drift_recovery_time
from float.visualization import plot, draw_selected_features, draw_top_features_with_reference, draw_concept_drifts

### Initialize Data Loader ###
scaler = SklearnScaler(scaler_obj=MinMaxScaler(), reset_after_drift=False)
data_loader = DataLoader(None, f'data/datasets/spambase.csv', target_col=0, scaler=scaler)

known_drifts = [round(data_loader.stream.n_samples * 0.2), round(data_loader.stream.n_samples * 0.4),
                round(data_loader.stream.n_samples * 0.6), round(data_loader.stream.n_samples * 0.8)]
batch_size = 10
feature_names = data_loader.stream.feature_names

concept_drift_detector_names = [  # 'ADWIN', 'EDDM', 'DDM_sk', 'DDM',
    'ERICS', 'Page Hinkley'
]  # Todo: Remove, and use class names instead?
concept_drift_detectors = [  # SkmultiflowChangeDetector(ADWIN(delta=0.6), reset_after_drift=False),
    # SkmultiflowChangeDetector(EDDM(), reset_after_drift=True),
    # SkmultiflowChangeDetector(DDM_scikit(), reset_after_drift=True),
    # DDM(reset_after_drift=True),
    ERICS(data_loader.stream.n_features),
    PageHinkley(reset_after_drift=True)]

cd_evaluator = dict()

for concept_drift_detector_name, concept_drift_detector in zip(concept_drift_detector_names, concept_drift_detectors):
    ### Initialize Predictor ###
    predictor = SkmultiflowClassifier(PerceptronMask(), data_loader.stream.target_values, reset_after_drift=True)  # todo: can we get rid of the target values parameter?
    pred_evaluator = PredictionEvaluator([zero_one_loss, mean_drift_performance_decay, mean_drift_recovery_time],  # noise_variability
                                         decay_rate=0.1,
                                         window_size=10,
                                         known_drifts=known_drifts,
                                         batch_size=batch_size,
                                         interval=10)

    ### Initialize Feature Selection ###
    ref_sample, _ = data_loader.stream.next_sample(50)
    data_loader.stream.reset()
    f_selector = FIRES(n_total_features=data_loader.stream.n_features,
                       n_selected_features=20,
                       classes=data_loader.stream.target_values,
                       reset_after_drift=False,
                       baseline='expectation',
                       ref_sample=ref_sample)
    fs_evaluator = FeatureSelectionEvaluator([nogueira_stability])

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
                                               feature_selector=f_selector,
                                               feature_selection_evaluator=fs_evaluator,
                                               concept_drift_detector=concept_drift_detector,
                                               change_detection_evaluator=cd_evaluator[concept_drift_detector_name],
                                               predictor=predictor,
                                               prediction_evaluator=pred_evaluator,
                                               batch_size=batch_size,
                                               max_n_samples=data_loader.stream.n_samples,
                                               known_drifts=known_drifts)
    prequential_pipeline.run()

"""
visualizer = Visualizer(
    [concept_drift_detector.drifts for concept_drift_detector in concept_drift_detectors],
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

visualizer = Visualizer(
    [pred_evaluator.result['accuracy_score']['measures'], pred_evaluator.result['accuracy_score']['mean_decay'], pred_evaluator.result['accuracy_score']['mean_window']],
    ['Accuracy Measures', 'Accuracy Decay', 'Accuracy Window'],
    'prediction')
visualizer.plot(
    plot_title=f'Metrics For Data Set spambase, Predictor Perceptron, Feature Selector FIRES',
    smooth_curve=[False, False, True])
plt.show()
"""
plot(measures=[pred_evaluator.result['mean_drift_performance_decay']['measures']],
     labels=['Perceptron'],
     measure_name='Drift Performance Decay',
     measure_type='prediction',
     smooth_curve=[False])
plt.show()

plot(measures=[pred_evaluator.result['mean_drift_recovery_time']['measures']],
     labels=['Perceptron'],
     measure_name='Drift Recovery Time',
     measure_type='prediction',
     smooth_curve=[False])
plt.show()

draw_top_features_with_reference(measures=[f_selector.selection],
                                 labels=['FIRES'],
                                 measure_type='feature_selection',
                                 feature_names=feature_names,
                                 fig_size=(15, 5))
plt.show()

draw_selected_features(measures=[f_selector.selection],
                       labels=['FIRES'],
                       measure_type='feature_selection',
                       layout=(1, 1),
                       fig_size=(10, 8))
plt.show()

plot(measures=[cd_evaluator['ERICS'].result['false_discovery_rate']['measures'], cd_evaluator['Page Hinkley'].result['false_discovery_rate']['measures']],
     labels=['ERICS', 'Page Hinkley'],
     measure_name='False Discovery Rate',
     measure_type='change_detection')
plt.show()

draw_concept_drifts(measures=[concept_drift_detectors[-2].drifts, concept_drift_detectors[-1].drifts],
                    labels=['ERICS', 'Page Hinkley'],
                    measure_type='change_detection',
                    data_stream=data_loader.stream,
                    known_drifts=known_drifts,
                    batch_size=batch_size)
plt.show()