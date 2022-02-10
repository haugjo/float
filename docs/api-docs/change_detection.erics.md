<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/erics.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `change_detection.erics`
ERICS Change Detection Method. 

The ERICS (Effective and Robust Identification of Concept Shift) change detector was proposed by: 

[1] HAUG, Johannes; KASNECI, Gjergji. Learning Parameter Distributions to Detect Concept Drift in Data Streams. In: 2020 25th International Conference on Pattern Recognition (ICPR). IEEE, 2021. S. 9452-9459. URL: [https://arxiv.org/pdf/2010.09388.pdf](https://arxiv.org/pdf/2010.09388.pdf)

The original source code can be obtained [here](https://github.com/haugjo/erics).

This module provides the ERICS implementation with a Probit base model for binary classification. The update rules for the Probit model are adopted from: 

[2] HAUG, Johannes, et al. Leveraging model inherent variable importance for stable online feature selection. In: Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2020. S. 1478-1502. URL: [https://dl.acm.org/doi/abs/10.1145/3394486.3403200](https://dl.acm.org/doi/abs/10.1145/3394486.3403200)

Copyright (C) 2022 Johannes Haug. 



---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/erics.py#L27"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ERICS`
ERICS Change Detector. 

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/erics.py#L29"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ERICS.__init__`

```python
__init__(
    n_param: int,
    window_mvg_average: int = 50,
    window_drift_detect: int = 50,
    beta: float = 0.0001,
    init_mu: int = 0,
    init_sigma: int = 1,
    epochs: int = 10,
    lr_mu: float = 0.01,
    lr_sigma: float = 0.01,
    reset_after_drift: bool = False
)
```

Inits the change detector. 



**Args:**

 - <b>`n_param`</b>:  The total number of parameters in the Probit model. This corresponds to the number of input features.  
 - <b>`window_mvg_average`</b>:  The window size for the moving average aggregation of KL divergence measures between the model parameter  distributions.  
 - <b>`window_drift_detect`</b>:  The window size that is used to compute the pairwise differences between subsequent measures of the  moving average. This window and information is used for the change detection.  
 - <b>`beta`</b>:  The scaling rate for the automatic update of the alpha-threshold, which is in turn applied to the  window_drift_detect to detect concept drift. 
 - <b>`init_mu`</b>:  The initial mean of the model parameter distributions. 
 - <b>`init_sigma`</b>:  The initial variance of the model parameter distributions. 
 - <b>`epochs`</b>:  The number of epochs per optimization iteration of the parameter distributions. 
 - <b>`lr_mu`</b>:  The learning rate for the gradient updates of the means. 
 - <b>`lr_sigma`</b>:  The learning rate for the gradient updates of the variances. 
 - <b>`reset_after_drift`</b>:  A boolean indicating if the change detector will be reset after a drift was detected. This is set to  False for ERICS, as this change detector does not need to be reset. 




---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/erics.py#L134"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ERICS.detect_change`

```python
detect_change() → bool
```

Detects global concept drift. 



**Returns:**
 
 - <b>`bool`</b>:  True, if a concept drift was detected, False otherwise. 

---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/erics.py#L142"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ERICS.detect_partial_change`

```python
detect_partial_change() → Tuple[bool, list]
```

Detects partial concept drift. 



**Returns:**
 
 - <b>`bool`</b>:  True, if at least one partial concept drift was detected, False otherwise. 
 - <b>`list`</b>:  Indices (i.e. relative positions in the feature vector) of input features with detected partial drift. 

---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/erics.py#L151"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ERICS.detect_warning_zone`

```python
detect_warning_zone() → bool
```

Detects a warning zone. 



**Notes:**

> ERICS does not raise warnings. 

---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/erics.py#L105"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ERICS.partial_fit`

```python
partial_fit(
    X: Union[numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]],
    y: Union[numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]]
)
```

Updates the change detector. 



**Args:**
 
 - <b>`X`</b>:  Batch of observations. 
 - <b>`y`</b>:  Batch of labels. 

---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/erics.py#L97"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ERICS.reset`

```python
reset()
```

Resets the change detector. 



**Notes:**

> ERICS need not be reset after a drift was detected. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
