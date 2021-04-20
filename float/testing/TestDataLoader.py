import unittest
from float.data.DataLoader import DataLoader
from skmultiflow.data.base_stream import Stream
import numpy as np


class TestDataLoader(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        self.data_loader = DataLoader(file_path='../data/datasets/spambase.csv', target_col=0)

    def test_evaluate_input(self):
        with self.assertRaises(AttributeError, msg='passing no inputs'):
            DataLoader()
        with self.assertRaises(AttributeError, msg='passing non .csv file'):
            DataLoader(file_path='../data/__init__.py')
        with self.assertRaises(AttributeError, msg='passing non int value for target_col'):
            DataLoader(file_path='../data/datasets/spambase.csv', target_col=None)
        with self.assertRaises(FileNotFoundError, msg='file not found'):
            DataLoader(file_path='../data/datasets/spam_base.csv')
        with self.assertRaises(ValueError, msg='non compatible .csv file'):
            DataLoader(file_path='../../docs/time_tracking.csv')

    def test_init(self):
        self.assertIsInstance(self.data_loader.stream, Stream, msg='attribute stream of type Stream')

    def test_get_data(self):
        n_samples = 3
        samples = self.data_loader.get_data(n_samples)
        self.assertIsInstance(samples, tuple, msg='returns tuple')
        self.assertIsInstance(samples[0], np.ndarray, msg='returns tuple of numpy arrays')
        self.assertIsInstance(samples[1], np.ndarray, msg='returns tuple of numpy arrays')
        self.assertEqual(samples[0].shape, (n_samples, self.data_loader.stream.n_features), msg='returns data in correct shape')
        self.assertEqual(samples[1].shape, (n_samples,), msg='returns target values in correct shape')
