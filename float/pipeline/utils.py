"""Pipeline Utils Module.

This module contains utility functions for the Pipeline Module. In particular, this module contains functionality to
validate provided attributes, update the console progress bar and print a final summary of the evaluation run.

Copyright (C) 2021 Johannes Haug

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import copy
import math
import numpy as np
import sys
from tabulate import tabulate
import time
from typing import TYPE_CHECKING
import warnings

import float.pipeline
from float.data.data_loader import DataLoader
from float.feature_selection import BaseFeatureSelector
from float.feature_selection.evaluation import FeatureSelectionEvaluator
from float.feature_selection.river import RiverFeatureSelector
from float.change_detection import BaseChangeDetector
from float.change_detection.evaluation import ChangeDetectionEvaluator
from float.change_detection.river import RiverChangeDetector
from float.change_detection.skmultiflow import SkmultiflowChangeDetector
from float.prediction import BasePredictor
from float.prediction.evaluation import PredictionEvaluator
from float.prediction.skmultiflow import SkmultiflowClassifier
from float.prediction.river import RiverClassifier

if TYPE_CHECKING:  # Only imports the below statements during type checking
    from float.pipeline.base_pipeline import BasePipeline


def validate_pipeline_attrs(pipeline: 'BasePipeline'):
    """Validates the input parameters and attributes of a pipeline obect.

    Args:
        pipeline: Pipeline object.

    Raises:
        AttributeError: If a crucial parameter to run the pipeline is missing or is invalid.
    """
    # Pipeline Validity Checks.
    if isinstance(pipeline, float.pipeline.DistributedFoldPipeline):
        if pipeline.validation_mode not in ['cross', 'split', 'bootstrap']:
            raise AttributeError('Please choose one of the validation modes "cross", "split", or "bootstrap" that are '
                                 'provided by the DistributedFoldPipeline.')

        if pipeline.n_parallel_instances < 2:
            warnings.warn('The DistributedFoldPipeline should use at least two instances of each provided '
                          'predictor for valid results. If you want to run a single instance, consider using the '
                          'PrequentialPipeline instead.')

    # Data Loader Validity Checks.
    if not isinstance(pipeline.data_loader, DataLoader):
        raise AttributeError('No valid DataLoader object was provided.')

    if pipeline.data_loader.stream.n_remaining_samples() < pipeline.data_loader.stream.n_samples:
        warnings.warn('The Data Loader object has not been reset. Float continues to run the pipeline on {}/{} '
                      'observations. If the specified known_drift positions do not account for the actual sample '
                      'size, float might return invalid performance measures.'.format(
                        pipeline.data_loader.stream.n_remaining_samples(),
                        pipeline.data_loader.stream.n_samples))

    # Predictor Validity Checks.
    for i, predictor in enumerate(pipeline.predictors):
        if predictor is None or not issubclass(type(predictor), BasePredictor):
            raise AttributeError(
                'At least one specified predictor is not a valid Predictor object.')

        if isinstance(predictor, RiverClassifier):
            if not predictor.can_mini_batch:
                warnings.warn('A specified RiverClassifier does not support batch processing. The batch '
                              'size is thus reset to 1 for all classifiers.')
                pipeline.batch_size = 1

        # Evaluate the first prediction evaluator object (note that all elements are equivalent)
        if pipeline.prediction_evaluators[i] is None or not isinstance(pipeline.prediction_evaluators[i],
                                                                       PredictionEvaluator):
            raise AttributeError(
                'Since a Predictor object was specified, a valid PredictionEvaluator object is also required '
                'but has not been provided.')

    # Change Detector Validity Checks.
    if pipeline.change_detector is not None:
        if not issubclass(type(pipeline.change_detector), BaseChangeDetector):
            raise AttributeError('No valid ChangeDetector object was provided.')

        if pipeline.change_detection_evaluator is None or not isinstance(pipeline.change_detection_evaluator,
                                                                         ChangeDetectionEvaluator):
            raise AttributeError(
                    'Since a ChangeDetector object was specified, a valid ChangeDetectionEvaluator object is also '
                    'required but has not been provided.')

        if pipeline.change_detector.error_based:
            if pipeline.predictors is None:
                raise AttributeError('An error-based ChangeDetector cannot be used without a valid Predictor '
                                     'object.')

            if isinstance(pipeline, float.pipeline.DistributedFoldPipeline):
                warnings.warn('An error-based ChangeDetector is being used in a DistributedFoldPipeline. Float '
                              'will use the prediction of the first predictor instance for the change detection.')

        if pipeline.n_pretrain is not None and pipeline.n_pretrain > 0:
            pipeline.change_detection_evaluator.n_pretrain = pipeline.n_pretrain
            pipeline.change_detection_evaluator.correct_known_drifts()
            pipeline.known_drifts = copy.copy(pipeline.change_detection_evaluator.known_drifts)
            warnings.warn('Known drift positions have been automatically corrected for the number of '
                          'observations used in pre-training, i.e. known_drift_position - n_pretrain')

    # Feature Selector Validity Checks.
    if pipeline.feature_selector is not None:
        if not issubclass(type(pipeline.feature_selector), BaseFeatureSelector):
            raise AttributeError('No valid FeatureSelector object was provided.')

        if pipeline.feature_selection_evaluator is None or not isinstance(pipeline.feature_selection_evaluator,
                                                                          FeatureSelectionEvaluator):
            raise AttributeError(
                    'Since a FeatureSelector object was specified, a valid FeatureSelectionEvaluator object is '
                    'also required but has not been provided.')

        if not pipeline.feature_selector.supports_multi_class and pipeline.data_loader.stream.n_classes > 2:
            raise AttributeError('The provided FeatureSelector does not support multiclass targets.')


def update_progress_bar(pipeline: 'BasePipeline'):
    """Updates the progress bar in the console after one training iteration.

    Args:
        pipeline: Pipeline object.
    """
    progress = math.ceil(pipeline.n_total / pipeline.n_max * 100)

    if pipeline.change_detector:
        n_detections = len(pipeline.change_detector.drifts)
        last_drift = pipeline.change_detector.drifts[-1] if n_detections > 0 else 0
        out_text = "[%-20s] %d%%, No. of detected drifts: %d, Last detected drift at t=%d." % (
            '=' * int(0.2 * progress), progress, n_detections, last_drift)
    else:
        out_text = "[%-20s] %d%%" % ('=' * int(0.2 * progress), progress)

    sys.stdout.write('\r')
    sys.stdout.write(out_text)
    sys.stdout.flush()


def print_evaluation_summary(pipeline: 'BasePipeline'):
    """Prints a summary of the given pipeline evaluation to the console.

    Args:
        pipeline: Pipeline object.
    """
    print('\n################################## SUMMARY ##################################')
    print('Evaluation has finished after {}s'.format(time.time() - pipeline.start_time))
    print('Data Set: {}'.format(pipeline.data_loader.path))
    print('The {} has processed {} instances, using batches of size {}.'.format(type(pipeline).__name__,
                                                                                pipeline.n_total,
                                                                                pipeline.batch_size))

    if None not in pipeline.predictors:
        print('-------------------------------------------------------------------------')
        print('*** Prediction ***')
        # Todo: account for distributed fold pipeline

        for predictor, prediction_evaluator in zip(pipeline.predictors, pipeline.prediction_evaluators):
            if isinstance(predictor, SkmultiflowClassifier) or isinstance(predictor, RiverClassifier):
                model_name = type(predictor).__name__.split('.')[-1] + '.' + type(predictor.model).__name__
            else:
                model_name = type(predictor).__name__.split('.')[-1]
            print('Model Name: {}'.format(model_name))

            tab_data = [['Avg. Test Comp. Time', np.nanmean(prediction_evaluator.testing_comp_times)],
                        ['Avg. Train Comp. Time',
                         np.nanmean(prediction_evaluator.training_comp_times)]]  # Aggregate data
            tab_data.extend([['Avg. {}'.format(key), value['mean'][-1]]
                             for key, value in prediction_evaluator.result.items()])
            print(tabulate(tab_data, headers=['Performance Measure', 'Value'], tablefmt='github'))
            print('------')

    if pipeline.change_detector is not None:
        print('-------------------------------------------------------------------------')
        print('*** Concept Drift Detection ***')

        if type(pipeline.change_detector) in [SkmultiflowChangeDetector, RiverChangeDetector]:
            model_name = type(pipeline.change_detector).__name__.split('.')[-1] + '.' + type(
                pipeline.change_detector.detector).__name__
        else:
            model_name = type(pipeline.change_detector).__name__.split('.')[-1]
        print('Model Name: {}'.format(model_name))

        if len(pipeline.change_detector.drifts) <= 5:
            print('Detected Global Drifts: {} ({} in total)'.format(str(pipeline.change_detector.drifts),
                                                                    len(pipeline.change_detector.drifts)))
        else:
            print('Detected Global Drifts: {}, ...] ({} in total)'.format(str(pipeline.change_detector.drifts[:5])[:-1],
                                                                          len(pipeline.change_detector.drifts)))

        tab_data = [['Avg. Comp. Time', np.nanmean(pipeline.change_detection_evaluator.comp_times)]]  # Aggregate data
        tab_data.extend([['Avg. {}'.format(key), value['mean']]
                         for key, value in pipeline.change_detection_evaluator.result.items()])
        print(tabulate(tab_data, headers=['Performance Measure', 'Value'], tablefmt='github'))

    if pipeline.feature_selector is not None:
        print('-------------------------------------------------------------------------')
        print('*** Online Feature Selection ***')

        if type(pipeline.feature_selector) is RiverFeatureSelector:
            model_name = type(pipeline.feature_selector).__name__.split('.')[-1] + '.' + type(
                pipeline.feature_selector.feature_selector).__name__
        else:
            model_name = type(pipeline.feature_selector).__name__.split('.')[-1]

        print('Model Name: {}'.format(model_name))
        print('Selected Features: {}/{}'.format(pipeline.feature_selector.n_selected_features,
                                                pipeline.feature_selector.n_total_features))

        tab_data = [['Avg. Comp. Time', np.nanmean(pipeline.feature_selection_evaluator.comp_times)]]  # Aggregate data
        tab_data.extend([['Avg. {}'.format(key), value['mean'][-1]]
                         for key, value in pipeline.feature_selection_evaluator.result.items()])
        print(tabulate(tab_data, headers=['Performance Measure', 'Value'], tablefmt='github'))

    print('#############################################################################')
