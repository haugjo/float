"""
The float.testing module includes testing functionality.
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

__all__ = ['TestDataLoader', 'TestVisualizer', 'TestSkmultiflowDriftDetector', 'TestPrequentialPipeline', 'TestSkmultiflowPerceptron',
           'TestCancelOutFeatureSelector', 'TestEFS', 'TestFIRES', 'TestFSDS', 'TestOFS', 'TestERICS', 'TestHoldoutPipeline']
