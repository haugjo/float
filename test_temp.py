# Import modules
import matplotlib.pyplot as plt
from sklearn.metrics import zero_one_loss, precision_score
from sklearn.preprocessing import MinMaxScaler
from skmultiflow.neural_networks.perceptron import PerceptronMask

# Import float modules
from float.data import DataLoader
from float.change_detection import ERICS
from float.change_detection.tornado import Adwin
from float.data.preprocessing import SklearnScaler
from float.pipeline import PrequentialPipeline, HoldoutPipeline, DistributedFoldPipeline
from float.change_detection.evaluation import ChangeDetectionEvaluator
from float.change_detection.evaluation.measures import mean_time_ratio, missed_detection_rate
from float.prediction.skmultiflow import SkmultiflowClassifier
from float.prediction.evaluation import PredictionEvaluator
from float.prediction.evaluation.measures import noise_variability

# - Create a data loader object -
data_loader = DataLoader(path='./float/data/datasets/electricity.csv',  # This path might have to be adjusted!
                         target_col=-1,
                         scaler=SklearnScaler(MinMaxScaler())
                         )

#test = data_loader.stream.next_sample(100)

# - Create the predictor object for each classifier that we want to compare.-
predictors = [SkmultiflowClassifier(model=PerceptronMask(), classes=data_loader.stream.target_values, reset_after_drift=True),
              SkmultiflowClassifier(model=PerceptronMask(), classes=data_loader.stream.target_values)
              ]

# - Create evaluator objects corresponding to each classifier. -
pred_evaluator = PredictionEvaluator(measure_funcs=[zero_one_loss, precision_score, noise_variability],
                                       decay_rate=0.1,
                                       window_size=25,
                                       zero_division=0,
                                       reference_measure=precision_score,
                                       n_samples=15)
# Change Detection
#change_detector = ERICS(n_param=data_loader.stream.n_features)
change_detector = Adwin()
known_drifts = [4707, (9396, 12500), 13570]
cd_evaluator = ChangeDetectionEvaluator(measure_funcs=[mean_time_ratio, missed_detection_rate],
                                        known_drifts=known_drifts,
                                        batch_size=10,
                                        n_total=data_loader.stream.n_samples,
                                        n_init_tolerance=25,
                                        n_delay=list(range(100, 1000)))

# - Run the Experiments for each classifier -
# - Create a prequential pipeline -
"""
pipeline = DistributedFoldPipeline(data_loader=data_loader,
                               predictor=predictors,
                               prediction_evaluator=pred_evaluator,
                               change_detector=change_detector,
                               change_detection_evaluator=cd_evaluator,
                               batch_size=100,
                               n_pretrain=200,
                               n_max=data_loader.stream.n_samples,  # We use all observations
                               random_state=0,
                                   label_delay_range=(5, 10),
                                   n_parallel_instances=5,
                                   validation_mode='bootstrap'
                           )
"""
pipeline = DistributedFoldPipeline(data_loader=data_loader)
pipeline.run()
