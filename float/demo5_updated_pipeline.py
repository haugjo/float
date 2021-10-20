from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.drift_detection.eddm import EDDM
from skmultiflow.drift_detection.ddm import DDM as DDM_scikit
from skmultiflow.neural_networks.perceptron import PerceptronMask
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, zero_one_loss, precision_score
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
from float.pipeline import HoldoutPipeline
from float.prediction.skmultiflow import SkmultiflowClassifier
from float.prediction.evaluation import PredictionEvaluator
from float.prediction.evaluation.measures import noise_variability, mean_drift_performance_deterioration, mean_drift_restoration_time
from float.visualization import plot, scatter, feature_selection_scatter, \
    concept_drift_detection_scatter, feature_selection_bar

### Initialize Data Loader ###
scaler = SklearnScaler(scaler_obj=MinMaxScaler(), reset_after_drift=False)
data_loader = DataLoader(path='data/datasets/spambase.csv', target_col=-1, scaler=scaler)

known_drifts = [round(data_loader.stream.n_samples * 0.2), round(data_loader.stream.n_samples * 0.4),
                round(data_loader.stream.n_samples * 0.6), round(data_loader.stream.n_samples * 0.8)]
batch_size = 10
feature_names = data_loader.stream.feature_names

change_detector_names = [  # 'ADWIN',
    'EDDM', #'DDM_sk',
    # 'DDM',
    #'ERICS', 'Page Hinkley'
]  # Todo: Remove, and use class names instead?
change_detectors = [  # SkmultiflowChangeDetector(ADWIN(delta=0.6), reset_after_drift=False),
    SkmultiflowChangeDetector(EDDM(), reset_after_drift=True),
    #SkmultiflowChangeDetector(DDM_scikit(), reset_after_drift=True),
    # SeqDrift2(reset_after_drift=True),
    ERICS(data_loader.stream.n_features),
    # PageHinkley(reset_after_drift=True)
    ]
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

for change_detector_name, change_detector in zip(change_detector_names, change_detectors):
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

            ### Initialize Change Detector ###
            cd_evaluator[change_detector_name] = ChangeDetectionEvaluator(measure_funcs=[detected_change_rate,
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
                                                       change_detector=change_detector,
                                                       change_detection_evaluator=cd_evaluator[
                                                           change_detector_name],
                                                       feature_selector=feature_selector,
                                                       feature_selection_evaluator=fs_evaluator[feature_selector_name],
                                                       batch_size=batch_size,
                                                       n_pretrain=0,
                                                       n_max=data_loader.stream.n_samples,
                                                       known_drifts=known_drifts,
                                                       estimate_memory_alloc=False,
                                                       random_state=1)
            prequential_pipeline.run()
            """
            holdout_pipeline = HoldoutPipeline(data_loader=data_loader,
                                               test_set=data_loader.get_data(10),
                                               feature_selector=feature_selector,
                                               feature_selection_evaluator=fs_evaluator[feature_selector_name],
                                               change_detector=change_detector,
                                               change_detection_evaluator=cd_evaluator[change_detector_name],
                                               predictor=predictor,
                                               prediction_evaluator=pred_evaluator[predictor_name],
                                               batch_size=batch_size,
                                               n_max=data_loader.stream.n_samples - 10,
                                               known_drifts=known_drifts,
                                               test_interval=50,
                                               test_replace_interval=20,
                                               random_state=0)
            holdout_pipeline.run()
            """

plot(measures=[pred_evaluator['Perceptron'].result['mean_drift_performance_deterioration']['measures']],
     variance_measures=[pred_evaluator['Perceptron'].result['mean_drift_performance_deterioration']['var']],
     legend_labels=['Perceptron'],
     y_label='Drift Performance Decay')
plt.show()

scatter(measures=[pred_evaluator['Perceptron'].result['mean_drift_performance_deterioration']['measures']],
        legend_labels=['Perceptron'],
        y_label='Drift Performance Decay')
plt.show()

plot(measures=[pred_evaluator['Perceptron'].result['mean_drift_restoration_time']['measures']],
     variance_measures=[pred_evaluator['Perceptron'].result['mean_drift_restoration_time']['var']],
     legend_labels=['Perceptron'],
     y_label='Drift Recovery Time')
plt.show()

feature_selection_bar(selected_features=[feature_selectors[0].selected_features_history],
                      model_names=['FIRES'],
                      feature_names=feature_names,
                      fig_size=(15, 5))
plt.show()

feature_selection_scatter(selected_features=feature_selectors[0].selected_features_history, fig_size=(10, 8))
plt.show()

concept_drift_detection_scatter(detected_drifts=[change_detectors[0].drifts, change_detectors[1].drifts],
                                model_names=['ERICS', 'Page Hinkley'],
                                n_samples=data_loader.stream.n_samples,
                                known_drifts=known_drifts,
                                batch_size=batch_size,
                                n_pretrain=0)
plt.show()
