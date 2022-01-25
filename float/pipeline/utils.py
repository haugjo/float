"""Pipeline Utils Module.

This module contains utility functions for the Pipeline Module. In particular, this module contains functionality to
print a summary of the evaluation run.

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

import numpy as np
from tabulate import tabulate
import time
from typing import TYPE_CHECKING

from float.feature_selection.river import RiverFeatureSelector
from float.change_detection.river import RiverChangeDetector
from float.change_detection.skmultiflow import SkmultiflowChangeDetector
from float.prediction.skmultiflow import SkmultiflowClassifier
from float.prediction.river import RiverClassifier

if TYPE_CHECKING:  # Only imports the below statements during type checking
    from float.pipeline.base_pipeline import BasePipeline


def print_summary(pipeline: 'BasePipeline'):
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

    if pipeline.feature_selector:
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

    if pipeline.change_detector:
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

    if pipeline.predictors:
        print('-------------------------------------------------------------------------')
        print('*** Prediction ***')

        if type(pipeline.predictors[0]) in [SkmultiflowClassifier, RiverClassifier]:
            model_name = type(pipeline.predictors[0]).__name__.split('.')[-1] + '.' + type(
                pipeline.predictors[0].model).__name__
        else:
            model_name = type(pipeline.predictors[0]).__name__.split('.')[-1]
        print('Model Name: {}'.format(model_name))

        tab_data = [['Avg. Test Comp. Time', np.nanmean(pipeline.prediction_evaluators[-1].testing_comp_times)],
                    ['Avg. Train Comp. Time',
                     np.nanmean(pipeline.prediction_evaluators[-1].training_comp_times)]]  # Aggregate data
        tab_data.extend([['Avg. {}'.format(key), value['mean'][-1]]
                         for key, value in pipeline.prediction_evaluators[-1].result.items()])  # Todo: deal with multiple predictors
        print(tabulate(tab_data, headers=['Performance Measure', 'Value'], tablefmt='github'))
    print('#############################################################################')
