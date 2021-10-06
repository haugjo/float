from skmultiflow.data.file_stream import FileStream
from skmultiflow.data.base_stream import Stream
from float.data.preprocessing import BaseScaler
import numpy as np


class DataLoader:
    """
    Serves as a wrapper for the skmultiflow Stream class.

    Attributes:
        stream (Stream): data stream object
        file_path (str): path to a .csv file containing the data
        target_col (int): index of the target column of the .csv file
        scaler (None | BaseScaler): identifier of a normalization/standardization technique
    """
    def __init__(self, stream=None, file_path=None, target_col=-1, scaler=None):
        """
        Receives either the path to a csv file (+ a target index) which is then mapped to a skmultiflow FileStream
        object OR a skmultiflow Stream object.

        Args:
            stream (skmultiflow.data.base_stream.Stream | None): data stream object
            file_path (str): path to a .csv file containing the data
            target_col (int): index of the target column of the .csv file
            scaler (None | BaseScaler): identifier of a normalization/standardization technique
        """
        self.stream = stream
        self.target_col = target_col
        self.file_path = file_path
        self.scaler = scaler
        self.__check_input()
        self.stream = stream if stream else FileStream(self.file_path, self.target_col)

    def get_data(self, n_samples):
        """
        Loads next batch of data from the Stream object. This is a wrapper for the skmultiflow
        Stream.next_sample(batch_size) function.

        Args:
            n_samples (int): number of samples to draw from data stream

        Returns:
            (np.ndarray, np.ndarray): data samples and targets
        """
        X, y = self.stream.next_sample(n_samples)

        if self.scaler:
            self.scaler.partial_fit(X)
            X = self.scaler.transform(X)

        return X, y

    def __check_input(self):
        """
        Evaluates the input data to check if it is in a valid format, i.e. if either a Stream object or a valid .csv
        file is provided. May be extended by further checks.

        Raises:
            AttributeError: if neither a valid skmultiflow Stream object nor a .csv file path is provided
            FileNotFoundError: if the provided file path does not exist
            ValueError: if the .csv cannot be converted to a skmultiflow FileStream
        """
        if not type(self.stream) is Stream:
            if not type(self.file_path) is str:
                raise AttributeError('Neither a valid skmultiflow Stream object nor a file path was provided.')
            elif not self.file_path.endswith('.csv'):
                raise AttributeError('Neither a valid skmultiflow Stream object nor a .csv file path was provided.')
            elif not type(self.target_col) is int:
                raise AttributeError('The parameter target_col needs to be an integer.')
            else:
                try:
                    FileStream(self.file_path, self.target_col)
                except FileNotFoundError:
                    raise FileNotFoundError('The file path you provided does not exist.')
                except ValueError:
                    raise ValueError('The .csv file cannot be converted to a skmultiflow FileStream.')
