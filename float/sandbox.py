# Here we may test code informally
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.neural_networks.perceptron import PerceptronMask
import matplotlib.pyplot as plt
from float import *

data_loader = data.DataLoader(None, 'data/datasets/spambase.csv', 0)
feature_selectors = [feature_selection.ofs.OFS(n_total_features=data_loader.stream.n_features, n_selected_features=10,
                                               nogueira_window_size=10),
                     feature_selection.fires.FIRES(n_total_features=data_loader.stream.n_features,
                                                   n_selected_features=10, classes=data_loader.stream.target_values,
                                                   nogueira_window_size=10)]
# adwin = ADWIN(delta=2)
# concept_drift_detector = concept_drift_detection.SkmultiflowDriftDetector(adwin)
concept_drift_detector = concept_drift_detection.erics.ERICS(data_loader.stream.n_features)
perceptron = PerceptronMask()
predictor = prediction.skmultiflow_perceptron.SkmultiflowPerceptron(perceptron, data_loader.stream.target_values)

for feature_selector in feature_selectors:
    pipeline.prequential_pipeline.PrequentialPipeline(data_loader, feature_selector, concept_drift_detector,
                                                      predictor, max_n_samples=data_loader.stream.n_samples).run()

visualizer = visualization.visualizer.Visualizer(
    [predictor.accuracy_scores, predictor.precision_scores, predictor.f1_scores, predictor.recall_scores], ['Accuracy', 'Precision', 'F1', 'Recall'],
    'prediction')
visualizer.scatter(plot_title='Predictor', layout=(2, 2))
plt.show()

visualizer = visualization.visualizer.Visualizer([feature_selectors[0].selection, feature_selectors[1].selection], ['OFS', 'FIRES'], 'feature_selection')
visualizer.draw_top_features(data_loader.stream.feature_names, layout=(1, 2), fig_size=(15, 5))
plt.show()
visualizer.draw_selected_features(layout=(2, 1), fig_size=(10, 8))
plt.show()
