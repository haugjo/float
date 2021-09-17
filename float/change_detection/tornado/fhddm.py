from float.change_detection.base_change_detector import BaseChangeDetector
import math


class FHDDM(BaseChangeDetector):
    """ Fast Hoeffding Drift Detection Method (FHDDM)

    Code adopted from https://github.com/alipsgh/tornado, please cite:
    The Tornado Framework
    By Ali Pesaranghader
    University of Ottawa, Ontario, Canada
    E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
    ---
    Paper: Pesaranghader, Ali, and Herna L. Viktor. "Fast hoeffding drift detection method for evolving data streams."
    Published in: Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Springer International Publishing, 2016.
    URL: https://link.springer.com/chapter/10.1007/978-3-319-46227-1_7

    Attributes:  # Todo: add attribute descriptions
    """
    def __init__(self, n=100, delta=0.000001):
        """ Initialize the concept drift detector

        Args:
            n (int):
            delta (float):
        """
        super().__init__(error_based=True)
        self.active_change = False

        self.__DELTA = delta
        self.__N = n
        self.__E = math.sqrt(math.log((1 / self.__DELTA), math.e) / (2 * self.__N))

        self.__WIN = []
        self.__MU_M = 0

    def reset(self):
        """ Resets the concept drift detector parameters.
        """
        self.__WIN.clear()
        self.__MU_M = 0

    def partial_fit(self, pr):
        """ Update the concept drift detector

        Args:
            pr (bool): indicator of correct prediction (i.e. pr=True) and incorrect prediction (i.e. pr=False)
        """
        self.active_change = False

        if len(self.__WIN) >= self.__N:
            self.__WIN.pop(0)
        self.__WIN.append(pr)

        if len(self.__WIN) >= self.__N:
            mu_t = self.__WIN.count(True) / self.__N
            if self.__MU_M < mu_t:
                self.__MU_M = mu_t
            self.active_change = (self.__MU_M - mu_t) > self.__E

    def detected_global_change(self):
        """ Checks whether global concept drift was detected or not.

        Returns:
            bool: whether global concept drift was detected or not.
        """
        return self.active_change

    def detected_warning_zone(self):
        pass

    def detected_partial_change(self):
        pass

    def get_length_estimation(self):
        pass
