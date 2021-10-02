from skmultiflow.neural_networks.perceptron import PerceptronMask
from sklearn.metrics import accuracy_score, zero_one_loss
from float import *

### Initialize Data Loader ###
data_loader = data.DataLoader(None, f'data/datasets/spambase.csv', target_col=0)

known_drifts = [round(data_loader.stream.n_samples * 0.2), round(data_loader.stream.n_samples * 0.4),
                round(data_loader.stream.n_samples * 0.6), round(data_loader.stream.n_samples * 0.8)]
batch_size = 10
feature_names = data_loader.stream.feature_names

### Initialize Feature Selector ###
fs_metrics = {
    'Nogueira Stability Measure': (
        feature_selection.base_feature_selector.FeatureSelector.get_nogueira_stability,
        {'n_total_features': data_loader.stream.n_features, 'nogueira_window_size': 10})
}

feature_selector = feature_selection.fires.FIRES(n_total_features=data_loader.stream.n_features, n_selected_features=10, classes=data_loader.stream.target_values, evaluation_metrics=fs_metrics)

### Initialize Concept Drift Detector ###
cdd_metrics = {
    'Delay': (
        change_detection.concept_drift_detector.ConceptDriftDetector.get_average_delay,
        {'known_drifts': known_drifts, 'batch_size': batch_size, 'max_n_samples': data_loader.stream.n_samples})
}
concept_drift_detector = change_detection.erics.ERICS(data_loader.stream.n_features, evaluation_metrics=cdd_metrics)

### Initialize Predictor ###
predictor = float.prediction.evaluation.skmultiflow.skmultiflow_perceptron.SkmultiflowPerceptron(PerceptronMask(),
                                                                                                 data_loader.stream.target_values,
                                                                                                 evaluation_metrics={'Accuracy': accuracy_score,
                                                                                        '0-1 Loss': zero_one_loss},
                                                                                                 decay_rate=0.5, window_size=5)

### Initialize and run Holdout Pipeline ###
test_size = 50
holdout_pipeline = pipeline.holdout_pipeline.HoldoutPipeline(data_loader, data_loader.get_data(test_size), 10, feature_selector,
                                                             concept_drift_detector, predictor, batch_size=batch_size,
                                                             max_n_samples=data_loader.stream.n_samples - test_size,
                                                             known_drifts=known_drifts)
holdout_pipeline.run()