from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.drift_detection.eddm import EDDM
from skmultiflow.drift_detection.ddm import DDM
from skmultiflow.neural_networks.perceptron import PerceptronMask
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, zero_one_loss
from float import *

### Initialize Data Loader ###
data_loader = data.DataLoader(None, f'data/datasets/spambase.csv', target_col=0)

known_drifts = [round(data_loader.stream.n_samples * 0.2), round(data_loader.stream.n_samples * 0.4),
                round(data_loader.stream.n_samples * 0.6), round(data_loader.stream.n_samples * 0.8)]
batch_size = 10
feature_names = data_loader.stream.feature_names

### Initialize Concept Drift Detector ###
cdd_metrics = {
        'Delay': (
            concept_drift_detection.concept_drift_detector.ConceptDriftDetector.get_average_delay,
            {'known_drifts': known_drifts, 'batch_size': batch_size, 'max_n_samples': data_loader.stream.n_samples}),
        'TPR': (
            concept_drift_detection.concept_drift_detector.ConceptDriftDetector.get_tpr,
            {'known_drifts': known_drifts, 'batch_size': batch_size, 'max_delay_range': 100}),
        'FDR': (
            concept_drift_detection.concept_drift_detector.ConceptDriftDetector.get_fdr,
            {'known_drifts': known_drifts, 'batch_size': batch_size, 'max_delay_range': 100}),
        'Precision': (
            concept_drift_detection.concept_drift_detector.ConceptDriftDetector.get_precision,
            {'known_drifts': known_drifts, 'batch_size': batch_size, 'max_delay_range': 100})
    }

concept_drift_detector_names = ['ADWIN', 'EDDM', 'DDM', 'ERICS']
concept_drift_detectors = [concept_drift_detection.SkmultiflowDriftDetector(ADWIN(delta=0.6), evaluation_metrics=cdd_metrics),
                           concept_drift_detection.SkmultiflowDriftDetector(EDDM(), evaluation_metrics=cdd_metrics),
                           concept_drift_detection.SkmultiflowDriftDetector(DDM(), evaluation_metrics=cdd_metrics),
                           concept_drift_detection.erics.ERICS(data_loader.stream.n_features, evaluation_metrics=cdd_metrics)]

### Initialize Predictor ###
predictor = prediction.skmultiflow_perceptron.SkmultiflowPerceptron(PerceptronMask(),
                                                                    data_loader.stream.target_values,
                                                                    evaluation_metrics={'Accuracy': accuracy_score,
                                                                                        '0-1 Loss': zero_one_loss},
                                                                    decay_rate=0.5, window_size=5)
for concept_drift_detector_name, concept_drift_detector in zip(concept_drift_detector_names, concept_drift_detectors):
    ### Initialize and run Prequential Pipeline ###
    prequential_pipeline = pipeline.prequential_pipeline.PrequentialPipeline(data_loader, None,
                                                                             concept_drift_detector,
                                                                             predictor,
                                                                             batch_size=batch_size,
                                                                             max_n_samples=data_loader.stream.n_samples,
                                                                             known_drifts=known_drifts)
    prequential_pipeline.run()

visualizer = visualization.visualizer.Visualizer(
    [concept_drift_detector.global_drifts for concept_drift_detector in concept_drift_detectors],
    concept_drift_detector_names, 'drift_detection')
visualizer.draw_concept_drifts(data_loader.stream, known_drifts, batch_size,
                               plot_title=f'Concept Drifts For Data Set spambase, Predictor Perceptron, Feature Selector FIRES')
plt.show()

visualizer = visualization.visualizer.Visualizer(
    [concept_drift_detector.evaluation['TPR'] for concept_drift_detector in
     concept_drift_detectors],
    concept_drift_detector_names, 'concept_drift_detection'
)
visualizer.plot(
    plot_title=f'Concept Drift True Positive Rate For Data Set spambase, Predictor Perceptron, Feature Selector FIRES')
plt.show()
