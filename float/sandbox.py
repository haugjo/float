from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.drift_detection.eddm import EDDM
from skmultiflow.drift_detection.ddm import DDM
from skmultiflow.neural_networks.perceptron import PerceptronMask
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, zero_one_loss
from float import *


def get_kdd_conceptdrift_feature_names():
    return ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
            "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
            "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
            "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
            "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
            "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
            "dst_host_srv_rerror_rate"]


data_set_names = ['spambase', 'adult_conceptdrift', 'har', 'kdd_conceptdrift']
feature_selector_names = ['OFS', 'FIRES']
predictor_names = ['Perceptron']
concept_drift_detector_names = ['ADWIN', 'EDDM', 'DDM', 'ERICS']

data_loaders = []
for file_name in data_set_names:
    data_loader = data.DataLoader(None, f'data/datasets/{file_name}.csv', 0)
    data_loaders.append(data_loader)

all_known_drifts = [[round(data_loader.stream.n_samples * 0.2), round(data_loader.stream.n_samples * 0.4),
                     round(data_loader.stream.n_samples * 0.6), round(data_loader.stream.n_samples * 0.8)] for
                    data_loader
                    in data_loaders]
batch_sizes = [10, 50, 10, 100]

for data_set_name, data_loader, batch_size, known_drifts in zip(data_set_names, data_loaders, batch_sizes,
                                                                all_known_drifts):
    feature_names = get_kdd_conceptdrift_feature_names() if data_set_name == 'kdd_conceptdrift' else data_loader.stream.feature_names

    fs_metrics = {'Nogueira Stability Measure': (
        feature_selection.feature_selector.FeatureSelector.get_nogueira_stability,
        {'n_total_features': data_loader.stream.n_features, 'nogueira_window_size': 10})}

    feature_selectors = [
        feature_selection.ofs.OFS(n_total_features=data_loader.stream.n_features, n_selected_features=10, evaluation_metrics=fs_metrics),
        feature_selection.fires.FIRES(n_total_features=data_loader.stream.n_features, n_selected_features=10, classes=data_loader.stream.target_values, evaluation_metrics=fs_metrics)]

    cdd_metrics = {
        'Delay': (
            change_detection.concept_drift_detector.ConceptDriftDetector.get_average_delay,
            {'known_drifts': known_drifts, 'batch_size': batch_size, 'max_n_samples': data_loader.stream.n_samples}),
        'TPR': (
            change_detection.concept_drift_detector.ConceptDriftDetector.get_tpr,
            {'known_drifts': known_drifts, 'batch_size': batch_size, 'max_delay_range': 100}),
        'FDR': (
            change_detection.concept_drift_detector.ConceptDriftDetector.get_fdr,
            {'known_drifts': known_drifts, 'batch_size': batch_size, 'max_delay_range': 100}),
        'Precision': (
            change_detection.concept_drift_detector.ConceptDriftDetector.get_precision,
            {'known_drifts': known_drifts, 'batch_size': batch_size, 'max_delay_range': 100})
    }
    concept_drift_detectors = [change_detection.SkmultiflowDriftDetector(ADWIN(delta=0.6), evaluation_metrics=cdd_metrics),
                               change_detection.SkmultiflowDriftDetector(EDDM(), evaluation_metrics=cdd_metrics),
                               change_detection.SkmultiflowDriftDetector(DDM(), evaluation_metrics=cdd_metrics),
                               change_detection.erics.ERICS(data_loader.stream.n_features, evaluation_metrics=cdd_metrics)]
    predictor = prediction.skmultiflow_perceptron.SkmultiflowPerceptron(PerceptronMask(),
                                                                        data_loader.stream.target_values,
                                                                        evaluation_metrics={'Accuracy': accuracy_score,
                                                                                            'Precision': (precision_score, {
                                                                                                'labels': data_loader.stream.target_values,
                                                                                                'average': 'weighted',
                                                                                                'zero_division': 0}),
                                                                                            'Recall': (recall_score, {
                                                                                                'labels': data_loader.stream.target_values,
                                                                                                'average': 'weighted',
                                                                                                'zero_division': 0}),
                                                                                            'F1 Score': (f1_score, {
                                                                                                'labels': data_loader.stream.target_values,
                                                                                                'average': 'weighted',
                                                                                                'zero_division': 0}),
                                                                                            '0-1 Loss': zero_one_loss},
                                                                        decay_rate=0.5, window_size=5)

    for feature_selector_name, feature_selector in zip(feature_selector_names, feature_selectors):
        for concept_drift_detector_name, concept_drift_detector in zip(concept_drift_detector_names,
                                                                       concept_drift_detectors):
            # prequential_pipeline = pipeline.prequential_pipeline.PrequentialPipeline(data_loader, feature_selector,
            #                                                                          concept_drift_detector,
            #                                                                          predictor,
            #                                                                          batch_size=batch_size,
            #                                                                          max_n_samples=data_loader.stream.n_samples,
            #                                                                          known_drifts=known_drifts)
            test_size = 50
            holdout_pipeline = pipeline.holdout_pipeline.HoldoutPipeline(data_loader, data_loader.get_data(test_size), 10, feature_selector,
                                                                         concept_drift_detector, predictor, batch_size=batch_size,
                                                                         max_n_samples=data_loader.stream.n_samples - test_size,
                                                                         known_drifts=known_drifts, run=True)

        visualizer = visualization.visualizer.Visualizer(
            [predictor.evaluation['Accuracy'], predictor.evaluation['Precision'], predictor.evaluation['F1 Score'], predictor.evaluation['Recall']],
            ['Accuracy', 'Precision', 'F1', 'Recall'],
            'prediction')
        visualizer.plot(plot_title=f'Metrics For Data Set {data_set_name}, Predictor {predictor_names[0]}, Feature Selector {feature_selector_name}', smooth_curve=False)
        plt.show()

        visualizer = visualization.visualizer.Visualizer(
            [predictor.evaluation_decay['Accuracy'], predictor.evaluation_window['Accuracy'], predictor.evaluation['Accuracy']],
            ['Accuracy Decay', 'Accuracy Window', 'Accuracy'],
            'prediction')
        visualizer.plot(
            plot_title=f'Metrics For Data Set {data_set_name}, Predictor {predictor_names[0]}, Feature Selector {feature_selector_name}',
            smooth_curve=[False, False, True])
        plt.show()

        visualizer = visualization.visualizer.Visualizer(
            [concept_drift_detector.global_drifts for concept_drift_detector in concept_drift_detectors],
            concept_drift_detector_names, 'drift_detection')
        visualizer.draw_concept_drifts(data_loader.stream, known_drifts, batch_size,
                                       plot_title=f'Concept Drifts For Data Set {data_set_name}, Predictor {predictor_names[0]}, Feature Selector {feature_selector_name}')
        plt.show()

        visualizer = visualization.visualizer.Visualizer(
            [concept_drift_detector.evaluation['TPR'] for concept_drift_detector in
             concept_drift_detectors],
            concept_drift_detector_names, 'change_detection'
        )
        visualizer.plot(
            plot_title=f'Concept Drift True Positive Rate For Data Set {data_set_name}, Predictor {predictor_names[0]}, Feature Selector {feature_selector_name}')
        plt.show()

    visualizer = visualization.visualizer.Visualizer(
        [feature_selectors[0].selection, feature_selectors[1].selection], feature_selector_names, 'feature_selection')
    visualizer.draw_top_features_with_reference(feature_names,
                                                f'Most Selected Features For Data Set {data_set_name}, Predictor {predictor_names[0]}',
                                                fig_size=(15, 5))
    plt.show()
    visualizer.draw_selected_features((2, 1),
                                      f'Selected Features At Each Time Step For Data Set {data_set_name}, Predictor {predictor_names[0]}',
                                      fig_size=(10, 8))
    plt.show()
    break
