from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.drift_detection.eddm import EDDM
from skmultiflow.drift_detection.ddm import DDM as DDM_scikit
from skmultiflow.neural_networks.perceptron import PerceptronMask
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, zero_one_loss
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer

from float.data import DataLoader
from float.data.preprocessing import SklearnScaler
from float.change_detection import ERICS
from float.change_detection.skmultiflow import SkmultiflowChangeDetector
from float.change_detection.tornado import PageHinkley, SeqDrift2
from float.change_detection.evaluation import ChangeDetectionEvaluator
from float.change_detection.evaluation.measures import detection_delay, detected_change_rate, \
    false_discovery_rate, time_between_false_alarms, mean_time_ratio, missed_detection_rate
from float.feature_selection.evaluation import FeatureSelectionEvaluator
from float.feature_selection.evaluation.measures import nogueira_stability
from float.feature_selection import FIRES
from float.pipeline import PrequentialPipeline
from float.prediction.skmultiflow import SkmultiflowClassifier
from float.prediction.evaluation import PredictionEvaluator
from float.prediction.evaluation.measures import noise_variability, mean_drift_performance_deterioration, mean_drift_restoration_time
from float.visualization import plot, selected_features_scatter, top_features_reference_bar, concept_drifts_scatter, spider_chart

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
concept_drift_detectors = [#SkmultiflowChangeDetector(ADWIN(delta=0.6), reset_after_drift=False),
    SkmultiflowChangeDetector(EDDM(), reset_after_drift=True),
    SkmultiflowChangeDetector(DDM_scikit(), reset_after_drift=True),
    #SeqDrift2(reset_after_drift=True),
    #ERICS(data_loader.stream.n_features),
    PageHinkley(reset_after_drift=True)]
cd_evaluator = dict()

pred_evaluator = dict()
predictor_names = ['Perceptron']
predictors = [SkmultiflowClassifier(PerceptronMask(), data_loader.stream.target_values, reset_after_drift=True)]  # todo: can we get rid of the target values parameter?

ref_sample, _ = data_loader.stream.next_sample(50)
data_loader.stream.reset()
fs_evaluator = dict()
feature_selector_names = ['FIRES']
feature_selectors = [FIRES(n_total_features=data_loader.stream.n_features,
                           n_selected_features=20,
                           classes=data_loader.stream.target_values,
                           reset_after_drift=False,
                           baseline='expectation',
                           ref_sample=ref_sample)]

for concept_drift_detector_name, concept_drift_detector in zip(concept_drift_detector_names, concept_drift_detectors):
    ### Initialize Predictor ###
    for predictor_name, predictor in zip(predictor_names, predictors):
        pred_evaluator[predictor_name] = PredictionEvaluator([accuracy_score, zero_one_loss, mean_drift_performance_deterioration, mean_drift_restoration_time, noise_variability],
                                                             decay_rate=0.1,
                                                             window_size=10,
                                                             known_drifts=known_drifts,
                                                             batch_size=batch_size,
                                                             interval=10)
        ### Initialize Feature Selection ###
        for feature_selector_name, feature_selector in zip(feature_selector_names, feature_selectors):
            fs_evaluator[feature_selector_name] = FeatureSelectionEvaluator([nogueira_stability])

            ### Initialize Concept Drift Detector ###
            cd_evaluator[concept_drift_detector_name] = ChangeDetectionEvaluator(measure_funcs=[detected_change_rate,
                                                                                                missed_detection_rate,
                                                                                                false_discovery_rate,
                                                                                                time_between_false_alarms,
                                                                                                detection_delay,
                                                                                                mean_time_ratio],
                                                                                 known_drifts=known_drifts,
                                                                                 batch_size=batch_size,
                                                                                 n_total=data_loader.stream.n_samples,
                                                                                 n_delay=list(range(100, 1000)),
                                                                                 n_init_tolerance=100)

            ### Initialize and run Prequential Pipeline ###
            prequential_pipeline = PrequentialPipeline(data_loader=data_loader,
                                                       predictor=predictor,
                                                       prediction_evaluator=pred_evaluator[predictor_name],
                                                       change_detector=concept_drift_detector,
                                                       change_detection_evaluator=cd_evaluator[
                                                           concept_drift_detector_name],
                                                       feature_selector=feature_selector,
                                                       feature_selection_evaluator=fs_evaluator[feature_selector_name],
                                                       batch_size=batch_size,
                                                       n_max=data_loader.stream.n_samples,
                                                       known_drifts=known_drifts)
            prequential_pipeline.run()

plot(measures=[pred_evaluator['Perceptron'].result['mean_drift_performance_deterioration']['measures']],
     variances=[pred_evaluator['Perceptron'].result['mean_drift_performance_deterioration']['var']],
     labels=['Perceptron'],
     measure_name='Drift Performance Decay',
     measure_type='prediction',
     smooth_curve=[False])
plt.show()

plot(measures=[pred_evaluator['Perceptron'].result['mean_drift_restoration_time']['measures']],
     variances=[pred_evaluator['Perceptron'].result['mean_drift_restoration_time']['var']],
     labels=['Perceptron'],
     measure_name='Drift Recovery Time',
     measure_type='prediction',
     smooth_curve=[False])
plt.show()

top_features_reference_bar(measures=[feature_selectors[0].selected_features_history],
                           labels=['FIRES'],
                           measure_type='feature_selection',
                           feature_names=feature_names,
                           fig_size=(15, 5))
plt.show()

selected_features_scatter(measures=[feature_selectors[0].selected_features_history],
                          labels=['FIRES'],
                          measure_type='feature_selection',
                          layout=(1, 1),
                          fig_size=(10, 8))
plt.show()

plot(measures=[cd_evaluator['ERICS'].result['false_discovery_rate']['measures'], cd_evaluator['Page Hinkley'].result['false_discovery_rate']['measures']],
     variances=[cd_evaluator['ERICS'].result['false_discovery_rate']['var'], cd_evaluator['Page Hinkley'].result['false_discovery_rate']['var']],
     labels=['ERICS', 'Page Hinkley'],
     measure_name='False Discovery Rate',
     measure_type='change_detection')
plt.show()

concept_drifts_scatter(measures=[concept_drift_detectors[-2].drifts, concept_drift_detectors[-1].drifts],
                       labels=['ERICS', 'Page Hinkley'],
                       measure_type='change_detection',
                       data_stream=data_loader.stream,
                       known_drifts=known_drifts,
                       batch_size=batch_size)
plt.show()

spider_chart(measures=[[pred_evaluator['Perceptron'].result['accuracy_score']['mean'][-1], pred_evaluator['Perceptron'].result['zero_one_loss']['mean'][-1],
                        pred_evaluator['Perceptron'].result['noise_variability']['mean'][-1], fs_evaluator['FIRES'].result['nogueira_stability']['mean'][-1],
                        cd_evaluator['ERICS'].result['missed_detection_rate']['mean'], cd_evaluator['ERICS'].result['false_discovery_rate']['mean']],
                       [pred_evaluator['Perceptron'].result['accuracy_score']['mean'][-1], pred_evaluator['Perceptron'].result['zero_one_loss']['mean'][-1],
                        pred_evaluator['Perceptron'].result['noise_variability']['mean'][-1], fs_evaluator['FIRES'].result['nogueira_stability']['mean'][-1],
                        cd_evaluator['Page Hinkley'].result['missed_detection_rate']['mean'], cd_evaluator['Page Hinkley'].result['false_discovery_rate']['mean']]
                       ],
             labels=['ERICS', 'Page Hinkley'],
             measure_names=['accuracy_score', 'zero_one_loss', 'noise_variability', 'nogueira_stability', 'missed_detection_rate', 'false_discovery_rate'])
plt.show()