<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/feature_selection/fires.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `feature_selection.fires`
FIRES Feature Selection Method. 

This module contains the Fast, Interpretable and Robust Evaluation and Selection of features (FIRES) with a Probit base model and normally distributed parameters as introduced by: HAUG, Johannes, et al. Leveraging model inherent variable importance for stable online feature selection. In: Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2020. S. 1478-1502. 

Copyright (C) 2022 Johannes Haug. 



---

<a href="https://github.com/haugjo/float/tree/main/float/feature_selection/fires.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FIRES`
FIRES feature selector. 

<a href="https://github.com/haugjo/float/tree/main/float/feature_selection/fires.py#L23"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `FIRES.__init__`

```python
__init__(
    n_total_features: int,
    n_selected_features: int,
    classes: list,
    mu_init: Union[int, numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, float, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]] = 0,
    sigma_init: Union[int, numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, float, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]] = 1,
    penalty_s: float = 0.01,
    penalty_r: float = 0.01,
    epochs: int = 1,
    lr_mu: float = 0.01,
    lr_sigma: float = 0.01,
    scale_weights: bool = True,
    reset_after_drift: bool = False,
    baseline: str = 'constant',
    ref_sample: Union[float, numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]] = 0
)
```

Inits the feature selector. 



**Args:**
 
 - <b>`n_total_features`</b>:  The total number of features. 
 - <b>`n_selected_features`</b>:  The number of selected features. 
 - <b>`classes`</b>:  A list of unique target values (class labels). mu_init:  Initial importance, i.e. mean of the parameter distribution. One may either set the initial values  separately per feature (by providing a vector), or use the same initial value for all features  (by providing a scalar). sigma_init:  Initial uncertainty, i.e. standard deviation of the parameter distribution. One may either set the  initial values separately per feature (by providing a vector), or use the same initial value for all  features (by providing a scalar). 
 - <b>`penalty_s`</b>:  Penalty factor in the optimization of weights w.r.t the uncertainty (corresponds to gamma_s in  the paper). 
 - <b>`penalty_r `</b>:  Penalty factor in the optimization of weights for the regularization (corresponds to gamma_r  in the paper). 
 - <b>`epochs`</b>:  Number of epochs in each update iteration. 
 - <b>`lr_mu`</b>:  Learning rate for the gradient update of the mean. 
 - <b>`lr_sigma`</b>:  Learning rate for the gradient update of the standard deviation. 
 - <b>`scale_weights`</b>:  If True, scale feature weights into the range [0,1]. If False, do not scale weights. 
 - <b>`reset_after_drift`</b>:  A boolean indicating if the change detector will be reset after a drift was detected. baseline:  A string identifier of the baseline method. The baseline is the value that we substitute non-selected  features with. This is necessary, because most online learning models are not able to handle arbitrary  patterns of missing data. ref_sample:  A sample used to compute the baseline. If the constant baseline is used, one needs to provide a single  float value. 




---

<a href="https://github.com/haugjo/float/tree/main/float/feature_selection/fires.py#L107"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `FIRES.reset`

```python
reset()
```

Resets the feature selector. 

---

<a href="https://github.com/haugjo/float/tree/main/float/feature_selection/fires.py#L89"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `FIRES.weight_features`

```python
weight_features(
    X: Union[numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]],
    y: Union[numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]]
)
```

Updates feature weights. 



**Args:**
 
 - <b>`X`</b>:  Array/matrix of observations. 
 - <b>`y`</b>:  Array of corresponding labels. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
