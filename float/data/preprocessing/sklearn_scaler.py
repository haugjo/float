from float.data.preprocessing import BaseScaler
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
import warnings


class SklearnScaler(BaseScaler):
    def __init__(self, scaler_obj, reset_after_drift=False):
        """ Wrapper for sklearn scaler functions

        Args:
            scaler_obj (obj): sklearn scaler object (e.g. MinMaxScaler)
            reset_after_drift (bool): indicates whether to reset the predictor after a drift was detected
        """
        super().__init__(reset_after_drift=reset_after_drift)
        self.scaler_obj = scaler_obj
        self._has_partial_fit = True
        self._must_be_fitted = False
        self.__validate()

    def partial_fit(self, X):
        """ Update/Fit the scaler

        Args:
            X (np.array): data sample
        """
        if self._must_be_fitted:
            self.scaler_obj.fit(X)
            self._must_be_fitted = False
        elif self._has_partial_fit:
            self.scaler_obj.partial_fit(X)

    def transform(self, X):
        """ Scale the given data sample

        Args:
            X (np.array): data sample

        Returns:
            np.array: scaled data sample
        """
        return self.scaler_obj.transform(X)

    def reset(self):
        """ Reset the sklearn scaler object
        We re-fit the scaler upon the next call to partial_fit
        """
        self._must_be_fitted = True

    def __validate(self):
        """ Validate the provided scaler object

        Raises: TypeError: if scaler object does neither have a partial_fit nor a fit function OR if it does not have
        a transform function
        """
        # Check if scaler object has a partial fit function
        partial_fit_func = getattr(self.scaler_obj, "partial_fit", None)
        if not callable(partial_fit_func):
            # Check if scaler object has a fit function
            fit_func = getattr(self.scaler_obj, "fit", None)
            if not callable(fit_func):
                raise TypeError("{} is not a valid sklearn scaler (missing 'fit' or 'partial_fit' function).".format(
                    type(self.scaler_obj).__name__))
            else:
                try:
                    self._has_partial_fit = False
                    check_is_fitted(self.scaler_obj)  # Check if scaler object has already been fitted
                except NotFittedError:
                    self._must_be_fitted = True
                    warnings.warn('Sklearn scaler object {} has not been fitted and will be fitted on the first batch '
                                  'of observations'.format(type(self.scaler_obj).__name__))
                    pass

        # Check if scaler object has a transform function
        transform_func = getattr(self.scaler_obj, "transform", None)
        if not callable(transform_func):
            raise TypeError("{} is not a valid sklearn scaler (missing 'transform' function).".format(
                type(self.scaler_obj).__name__))
