# Here we may test code informally
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.neural_networks.perceptron import PerceptronMask
import matplotlib.pyplot as plt
from float import *

data_loader = data.DataLoader(None, 'data/datasets/spambase.csv', 0)
feature_selector = feature_selection.ofs.OFS(n_total_features=data_loader.stream.n_features, n_selected_features=10,
                                             nogueira_window_size=10)
adwin = ADWIN(delta=2)
concept_drift_detector = concept_drift_detection.SkmultiflowDriftDetector(adwin)
perceptron = PerceptronMask()
predictor = prediction.skmultiflow_perceptron.SkmultiflowPerceptron(perceptron, data_loader.stream.target_values)

pipeline = pipeline.prequential_pipeline.PrequentialPipeline(data_loader, feature_selector, concept_drift_detector,
                                                             predictor, max_n_samples=data_loader.stream.n_samples)
pipeline.run()

visualizer = visualization.visualizer.Visualizer([predictor.accuracy_scores, predictor.precision_scores, predictor.f1_scores], ['Accuracy', 'Precision', 'F1'], 'prediction')
visualizer.scatter(plot_title='Predictor')
plt.show()

visualizer = visualization.visualizer.Visualizer(feature_selector.selection, [], 'feature_selection')
visualizer.draw_top_features(data_loader.stream.feature_names)
plt.show()
visualizer.draw_selected_features()
plt.show()
