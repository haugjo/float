"""Testing Module.

This module contains unit tests to validate all central functionality of the float package.

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
from .test_visualization_bar import TestBar
from .test_feature_selection_cancel_out import TestCancelOutFeatureSelector
from .test_change_detection_evaluator import TestChangeDetectionEvaluator
from .test_visualization_concept_drift_detection_scatter import TestConceptDriftDetectionScatter
from float.unit_tests.test_data_loader import TestDataLoader
from .test_feature_selection_efs import TestEFS
from .test_change_detection_erics import TestERICS
from .test_feature_selection_fires import TestFIRES
from .test_visualization_feature_selection_bar import TestFeatureSelectionBar
from .test_feature_selection_evaluator import TestFeatureSelectionEvaluator
from .test_visualization_feature_selection_scatter import TestFeatureSelectionScatter
from .test_feature_selection_fsds import TestFSDS
from .test_pipeline_holdout import TestHoldoutPipeline
from .test_feature_selection_ofs import TestOFS
from .test_visualization_plot import TestPlot
from .test_prediction_evaluator import TestPredictionEvaluator
from .test_pipeline_prequential import TestPrequentialPipeline
from .test_change_detection_river import TestRiverChangeDetector
from .test_prediction_river_classifier import TestRiverClassifier
from .test_visualization_scatter import TestScatter
from float.unit_tests.test_data_sklearn_scaler import TestSklearnScaler
from .test_prediction_skmultiflow_classifier import TestSkmultiflowClassifier
from .test_change_detection_skmultiflow import TestSkmultiflowChangeDetector
from .test_visualization_spider_chart import TestSpiderChart

__all__ = ['TestBar',  'TestCancelOutFeatureSelector',  'TestChangeDetectionEvaluator', 'TestConceptDriftDetectionScatter',
           'TestDataLoader', 'TestEFS', 'TestERICS', 'TestFIRES', 'TestFeatureSelectionBar', 'TestFeatureSelectionEvaluator',
           'TestFeatureSelectionScatter', 'TestFSDS', 'TestHoldoutPipeline', 'TestOFS', 'TestPlot', 'TestPredictionEvaluator',
           'TestPrequentialPipeline', 'TestRiverChangeDetector', 'TestRiverClassifier', 'TestScatter', 'TestSklearnScaler',
           'TestSkmultiflowClassifier', 'TestSkmultiflowChangeDetector', 'TestSpiderChart']
