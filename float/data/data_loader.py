"""Data Loader Module.

This module encapsulates functionality to load and preprocess input data. The data loader class uses the
scikit-multiflow Stream class to simulate streaming data.

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
from numpy.typing import ArrayLike
from skmultiflow.data.file_stream import FileStream
from skmultiflow.data.base_stream import Stream
from typing import Optional, Tuple

from float.data.preprocessing import BaseScaler


class DataLoader:
    """Data Loader Class

    The data loader class is responsible to sample and pre-process (i.e. normalize) input data, thereby simulating a
    data stream. The data loader uses a skmultiflow Stream object to generate or load streaming data.

    Attributes:
        stream (Stream | None): A scikit-multiflow data stream object.
        file_path (str | None): The path to a .csv file containing the training data set.
        target_col (int): The index of the target column in the training data.
        scaler (BaseScaler | None): A scaler object used to normalize/standardize sampled instances.
    """
    def __init__(self, stream: Optional[Stream] = None, file_path: Optional[str] = None, target_col: int = -1,
                 scaler: Optional[BaseScaler] = None):
        """Inits the data loader.

        The data loader init function must receive either one of the following inputs:
        - the path to a .csv file (+ a target index), which is then mapped to a skmultiflow FileStream object.
        - a valid scikit multiflow Stream object.

        Args:
            stream: A scikit-multiflow data stream object.
            file_path: The path to a .csv file containing the training data set.
            target_col: The index of the target column in the training data.
            scaler: A scaler object used to normalize/standardize sampled instances.
        """
        self.stream = stream
        self.file_path = file_path
        self.target_col = target_col
        self.scaler = scaler

        self._check_input()
        self.stream = stream if stream else FileStream(self.file_path, self.target_col)

    def get_data(self, n_batch: int) -> Tuple[ArrayLike, ArrayLike]:
        """Loads a batch from the stream object.

        Args:
            n_batch: Number of samples to load from the data stream object.

        Returns:
            Tuple[ArrayLike, ArrayLike]: The sampled observations and corresponding targets.
        """
        X, y = self.stream.next_sample(batch_size=n_batch)

        if self.scaler:
            self.scaler.partial_fit(X)
            X = self.scaler.transform(X)

        return X, y

    def _check_input(self):
        """Validates the input.

        This function checks whether the constructor either received a valid path to a .csv file or a valid
        scikit-multiflow Stream object.

        Raises:
            AttributeError: If neither a valid scikit multiflow Stream object nor a .csv file path is provided.
            FileNotFoundError: If the provided file path does not exist.
            ValueError: If the .csv cannot be converted to a scikit multiflow FileStream.
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
