import unittest
from float.data.data_loader import DataLoader
from skmultiflow.data.base_stream import Stream
import numpy as np


class TestDataLoader(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        self.data_loader = DataLoader(path='../data/datasets/spambase.csv', target_col=-1)

    def test_init(self):
        self.assertIsInstance(self.data_loader.stream, Stream, msg='attribute stream is initialized correctly')
        with self.assertRaises(AttributeError, msg='AttributeError when passing no inputs'):
            DataLoader()
        with self.assertRaises(AttributeError, msg='AttributeError when passing non .csv file'):
            DataLoader(path='../data/__init__.py')
        with self.assertRaises(AttributeError, msg='AttributeError when passing non int value for target_col'):
            DataLoader(path='../data/datasets/spambase.csv', target_col=None)
        with self.assertRaises(FileNotFoundError, msg='FileNotFoundError when file not found'):
            DataLoader(path='../data/datasets/spam_base.csv')

    def test_get_data(self):
        n_samples = 3
        samples = self.data_loader.get_data(n_samples)
        self.assertIsInstance(samples, tuple, msg='get_data() returns tuple')
        self.assertIsInstance(samples[0], np.ndarray, msg='get_data() returns tuple of numpy arrays')
        self.assertIsInstance(samples[1], np.ndarray, msg='get_data() returns tuple of numpy arrays')
        self.assertEqual(samples[0].shape, (n_samples, self.data_loader.stream.n_features), msg='get_data() returns data in correct shape')
        self.assertEqual(samples[1].shape, (n_samples,), msg='get_data() returns target values in correct shape')
