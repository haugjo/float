"""Data Loader.

This module encapsulates functionality to load and preprocess input data. The data loader class uses the
scikit-multiflow Stream class to simulate streaming data.

Copyright (C) 2022 Johannes Haug.
"""
from numpy.typing import ArrayLike
from skmultiflow.data.file_stream import FileStream
from skmultiflow.data.base_stream import Stream
from typing import Optional, Tuple

from float.data.preprocessing import BaseScaler


class DataLoader:
    """Data Loader Class.

    The data loader class is responsible to sample and pre-process (i.e. normalize) input data, thereby simulating a
    data stream. The data loader uses a skmultiflow Stream object to generate or load streaming data.

    Attributes:
        path (str | None): The path to a .csv file containing the training data set.
        stream (Stream | None): A scikit-multiflow data stream object.
        target_col (int): The index of the target column in the training data.
        scaler (BaseScaler | None): A scaler object used to normalize/standardize sampled instances.
    """
    def __init__(self,
                 path: Optional[str] = None,
                 stream: Optional[Stream] = None,
                 target_col: int = -1,
                 scaler: Optional[BaseScaler] = None):
        """Inits the data loader.

        The data loader init function must receive either one of the following inputs:
        - the path to a .csv file (+ a target index), which is then mapped to a skmultiflow FileStream object.
        - a valid scikit multiflow Stream object.

        Args:
            path: The path to a .csv file containing the training data set.
            stream: A scikit-multiflow data stream object.
            target_col: The index of the target column in the training data.
            scaler: A scaler object used to normalize/standardize sampled instances.
        """
        self.stream = stream
        self.path = path
        self.target_col = target_col
        self.scaler = scaler

        self._validate()
        self.stream = stream if stream else FileStream(self.path, self.target_col)

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

    def _validate(self):
        """Validates the input.

        This function checks whether the constructor either received a valid path to a .csv file or a valid
        scikit-multiflow Stream object.

        Raises:
            AttributeError: If neither a valid scikit multiflow Stream object nor a .csv file path is provided.
            FileNotFoundError: If the provided file path does not exist.
            ValueError: If the .csv cannot be converted to a scikit multiflow FileStream.
        """
        if not isinstance(self.stream, Stream):
            if not isinstance(self.path, str):
                raise AttributeError("Neither a valid scikit-multiflow Stream object nor a file path was provided.")
            elif not self.path.endswith('.csv'):
                raise AttributeError("Neither a valid scikit-multiflow Stream object nor a .csv-file path was provided.")
            elif not isinstance(self.target_col, int):
                raise AttributeError("The parameter target_col needs to be an integer.")
            else:
                try:
                    FileStream(self.path, self.target_col)
                except FileNotFoundError:
                    raise FileNotFoundError("The file path provided does not exist.")
                except ValueError:
                    raise ValueError("The .csv file cannot be converted to a scikit-multiflow FileStream.")

        if self.scaler is not None and not isinstance(self.scaler, BaseScaler):
            raise AttributeError("The provided scaler is not a valid Scaler object.")
