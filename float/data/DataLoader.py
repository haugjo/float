class DataLoader:
    """
    Description
    """
    def __init__(self):
        """
        Description
        """
        raise NotImplementedError

    def get_data(self, n_samples):
        """
        Description

        :param (int) n_samples: no. of samples to draw from data stream
        :return: samples
        :rtype np.array
        """
        raise NotImplementedError

    def _evaluate_input(self):
        """
        Description

        :return:
        """
        raise NotImplementedError
