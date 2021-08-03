from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.neural_networks.perceptron import PerceptronMask
import matplotlib.pyplot as plt
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


data_loaders = []
for file_name in ['spambase', 'adult_conceptdrift', 'har', 'kdd_conceptdrift']:
    data_loader = data.DataLoader(None, f'data/datasets/{file_name}.csv', 0)
    data_loaders.append(data_loader)

known_drifts = [[round(data_loader.stream.n_samples * 0.2), round(data_loader.stream.n_samples * 0.4),
                 round(data_loader.stream.n_samples * 0.6), round(data_loader.stream.n_samples * 0.8)] for data_loader
                in data_loaders]
batch_sizes = [10, 50, 10, 100]

pipelines = []
for i, data_loader in enumerate(data_loaders):
    feature_names = get_kdd_conceptdrift_feature_names() if i == 3 else data_loader.stream.feature_names
    feature_selectors = [
        feature_selection.ofs.OFS(n_total_features=data_loader.stream.n_features, n_selected_features=10,
                                  nogueira_window_size=10),
        feature_selection.fires.FIRES(n_total_features=data_loader.stream.n_features,
                                      n_selected_features=10, classes=data_loader.stream.target_values,
                                      nogueira_window_size=10)]

    concept_drift_detectors = [concept_drift_detection.SkmultiflowDriftDetector(ADWIN()),
                               concept_drift_detection.erics.ERICS(data_loader.stream.n_features)]
    predictor = prediction.skmultiflow_perceptron.SkmultiflowPerceptron(PerceptronMask(),
                                                                        data_loader.stream.target_values)
    for feature_selector in feature_selectors:
        for concept_drift_detector in concept_drift_detectors:
            prequential_pipeline = pipeline.prequential_pipeline.PrequentialPipeline(data_loader, feature_selector,
                                                                                     concept_drift_detector,
                                                                                     predictor,
                                                                                     batch_size=batch_sizes[i],
                                                                                     max_n_samples=data_loader.stream.n_samples)
            prequential_pipeline.run()
            pipelines.append(prequential_pipeline)

    visualizer = visualization.visualizer.Visualizer(
        [predictor.accuracy_scores, predictor.precision_scores, predictor.f1_scores, predictor.recall_scores],
        ['Accuracy', 'Precision', 'F1', 'Recall'],
        'prediction')
    visualizer.plot(plot_title='Predictor')
    plt.show()

    visualizer = visualization.visualizer.Visualizer(
        [feature_selectors[0].selection, feature_selectors[1].selection], ['OFS', 'FIRES'], 'feature_selection')
    print('DATA FEATURES:', feature_names)
    visualizer.draw_top_features_with_reference(feature_names, fig_size=(15, 5))
    plt.show()
    visualizer.draw_selected_features(layout=(2, 1), fig_size=(10, 8))
    plt.show()

visualizer = visualization.visualizer.Visualizer(
    [concept_drift_detectors[0].global_drifts, concept_drift_detectors[1].global_drifts], ['ERICS'], 'drift_detection')
visualizer.draw_concept_drifts([data_loader.stream for data_loader in data_loaders], known_drifts, batch_sizes,
                               ['spambase', 'adult', 'har', 'kdd'], ['adwin', 'erics'])
