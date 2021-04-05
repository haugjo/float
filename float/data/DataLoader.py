from skmultiflow.data.file_stream import FileStream
from skmultiflow.data.base_stream import Stream


class DataLoader:
    """
    Serves as a wrapper for the scikit-multiflow Stream module.

    Attributes:
        stream (skmultiflow.data.base_stream.Stream): data stream object
        file_path (str): path to a .csv file containing the data
        target_col (int): index of the target column of the .csv file
    """
    def __init__(self, stream=None, file_path=None, target_col=-1):
        """
        Receives either the path to a csv file (+ a target index) which is then mapped to a skmultiflow FileStream
        object OR a skmultiflow Stream object.

        Args:
            stream (skmultiflow.data.base_stream.Stream): data stream object
            file_path (str): path to a .csv file containing the data
            target_col (int): index of the target column of the .csv file

        """
        self.stream = stream
        self.target_col = target_col
        self.file_path = file_path
        if not self._evaluate_input():
            raise AttributeError
        self.stream = stream if stream else FileStream(self.file_path, self.target_col)

    def get_data(self, n_samples):
        """
        Loads next batch of data from the Stream object. This is a wrapper for the skmultiflow
        Stream.next_sample(batch_size) function.

        Args:
            n_samples (int): number of samples to draw from data stream

        Returns:
            np.array: data samples
        """
        return self.stream.next_sample(n_samples)

    def _evaluate_input(self):
        """
        Evaluates the input data to check if it is in a valid format, i.e. if either a Stream object or a valid .csv
        file is provided. May be extended by further checks.

        Returns:
            bool: True if a valid input is provided, False otherwise
        """
        if type(self.stream) is Stream:
            return True
        if type(self.file_path) is str and type(self.target_col) is int:
            if self.file_path.endswith('.csv'):
                try:
                    FileStream(self.file_path, self.target_col)
                    return True
                except (FileNotFoundError, ValueError):
                    pass
        return False
