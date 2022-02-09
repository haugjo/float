"""Testing Module.

This module contains unit tests to validate all central functionality of the float package.

Copyright (C) 2022 Johannes Haug.
"""
from .test_visualization_bar import TestBar
from .test_feature_selection_cancel_out import TestCancelOutFeatureSelector
from .test_change_detection_evaluator import TestChangeDetectionEvaluator
from .test_visualization_change_detection_scatter import TestConceptDriftDetectionScatter
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
