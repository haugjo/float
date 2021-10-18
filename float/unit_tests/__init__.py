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
from .test_data_loader import TestDataLoader
from .test_visualizer import TestVisualizer
from .test_skmultiflow_drift_detector import TestSkmultiflowDriftDetector
from .test_prequential_pipeline import TestPrequentialPipeline
from .test_skmultiflow_perceptron import TestSkmultiflowPerceptron
from .test_cancel_out import TestCancelOutFeatureSelector
from .test_efs import TestEFS
from .test_fires import TestFIRES
from .test_fsds import TestFSDS
from .test_ofs import TestOFS
from .test_erics import TestERICS
from .test_holdout_pipeline import TestHoldoutPipeline
from .test_prediction_evaluator import TestPredictionEvaluator
from .test_feature_selection_evaluator import TestFeatureSelectionEvaluator
from .test_change_detection_evaluator import TestChangeDetectionEvaluator

__all__ = ['TestDataLoader', 'TestVisualizer', 'TestSkmultiflowDriftDetector', 'TestPrequentialPipeline', 'TestSkmultiflowPerceptron',
           'TestCancelOutFeatureSelector', 'TestEFS', 'TestFIRES', 'TestFSDS', 'TestOFS', 'TestERICS', 'TestHoldoutPipeline']
