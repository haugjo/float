from float.data.data_loader import DataLoader
import numpy as np
from skmultiflow.data.base_stream import Stream
from skmultiflow.data import FileStream
import unittest


class TestDataLoader(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)

    def test_init(self):
        data_loader = DataLoader(path='../datasets/spambase.csv', target_col=-1)
        self.assertIsInstance(data_loader.stream, Stream, msg='Stream is initialized correctly from path')

        skm_stream = FileStream('../datasets/spambase.csv', -1)
        data_loader = DataLoader(stream=skm_stream, target_col=-1)
        self.assertIsInstance(data_loader.stream, Stream, msg='Stream is initialized correctly from FileStream')

        with self.assertRaises(AttributeError, msg='AttributeError when passing no inputs'):
            DataLoader()
        with self.assertRaises(AttributeError, msg='AttributeError when passing non .csv file'):
            DataLoader(path='../float/data/__init__.py')
        with self.assertRaises(AttributeError, msg='AttributeError when passing non int value for target_col'):
            DataLoader(path='../datasets/spambase.csv', target_col=None)
        with self.assertRaises(FileNotFoundError, msg='FileNotFoundError when file not found'):
            DataLoader(path='../data/datasets/spam_base.csv')

    def test_get_data(self):
        data_loader = DataLoader(path='../datasets/spambase.csv', target_col=-1)
        samples = data_loader.get_data(data_loader.stream.n_samples)
        self.assertIsInstance(samples, tuple, msg='get_data() returns tuple')
        self.assertIsInstance(samples[0], np.ndarray, msg='get_data() returns tuple of numpy arrays')
        self.assertIsInstance(samples[1], np.ndarray, msg='get_data() returns tuple of numpy arrays')
        self.assertEqual(samples[0].shape, (data_loader.stream.n_samples, data_loader.stream.n_features),
                         msg='get_data() returns data in correct shape')
        self.assertEqual(samples[1].shape, (data_loader.stream.n_samples,),
                         msg='get_data() returns target values in correct shape')
