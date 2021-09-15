from skmultiflow.neural_networks.perceptron import PerceptronMask
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, zero_one_loss
from float import *

### Initialize Data Loader ###
data_loader = data.DataLoader(None, f'data/datasets/spambase.csv', target_col=0)

known_drifts = [round(data_loader.stream.n_samples * 0.2), round(data_loader.stream.n_samples * 0.4),
                round(data_loader.stream.n_samples * 0.6), round(data_loader.stream.n_samples * 0.8)]
batch_size = 10
feature_names = data_loader.stream.feature_names

### Initialize Predictor ###
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

### Initialize and run Prequential Pipeline ###
prequential_pipeline = pipeline.prequential_pipeline.PrequentialPipeline(data_loader, None,
                                                                         None,
                                                                         predictor,
                                                                         batch_size=batch_size,
                                                                         max_n_samples=data_loader.stream.n_samples,
                                                                         known_drifts=known_drifts)
prequential_pipeline.run()

### Predictor plots ###
visualizer = visualization.visualizer.Visualizer(
    [predictor.evaluation['Accuracy'], predictor.evaluation['Precision'],
     predictor.evaluation['F1 Score'], predictor.evaluation['Recall']],
    ['Accuracy', 'Precision', 'F1', 'Recall'],
    'prediction')
visualizer.plot(plot_title=f'Metrics For Data Set spambase, Predictor Perceptron, Feature Selector FIRES', smooth_curve=False)
plt.show()

visualizer = visualization.visualizer.Visualizer(
    [predictor.evaluation_decay['Accuracy'], predictor.evaluation_window['Accuracy'], predictor.evaluation['Accuracy']],
    ['Accuracy Decay', 'Accuracy Window', 'Accuracy'],
    'prediction')
visualizer.plot(
    plot_title=f'Metrics For Data Set spambase, Predictor Perceptron, Feature Selector FIRES',
    smooth_curve=[False, False, True])
plt.show()
