# class *Predictor*
#### Description
Abstract base class which serves as both a wrapper for scikit-multiflow predictive models and a constructor for user-defined predictive models.

#### Attributes
name
: string

#### Functions
fit
: Fits the model.

partial_fit
: Partially (incrementally) fits the model.

predict
: Predicts target values for the passed data.

predict_proba
: Estimates the probability for probabilistic/bayesian regressors.

score
: Returns the accuracy based on the given test samples and true values.