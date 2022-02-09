<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/feature_selection/efs.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `feature_selection.efs`
Extremal Feature Selection Method. 

This module contains the Extremal Feature Selection model introduced by: CARVALHO, Vitor R.; COHEN, William W. Single-pass online learning: Performance, voting schemes and online feature selection. In: Proceedings of the 12th ACM SIGKDD international conference on Knowledge discovery and data mining. 2006. S. 548-553. 

Copyright (C) 2022 Johannes Haug. 



---

<a href="https://github.com/haugjo/float/tree/main/float/feature_selection/efs.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `EFS`
Extremal feature selector. 

This feature selection algorithm uses the weights of a Modified Balanced Winnow classifier. 

<a href="https://github.com/haugjo/float/tree/main/float/feature_selection/efs.py#L23"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `EFS.__init__`

```python
__init__(
    n_total_features: int,
    n_selected_features: int,
    u: Optional[numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]] = None,
    v: Optional[numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]] = None,
    theta: float = 1,
    M: float = 1,
    alpha: float = 1.5,
    beta: float = 0.5,
    reset_after_drift: bool = False,
    baseline: str = 'constant',
    ref_sample: Union[float, numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]] = 0
)
```

Inits the feature selector. 



**Args:**
 
 - <b>`n_total_features`</b>:  The total number of features. 
 - <b>`n_selected_features`</b>:  The number of selected features. 
 - <b>`u`</b>:  Initial positive model weights of the Winnow algorithm. 
 - <b>`v`</b>:  Initial negative model weights of the Winnow algorithm. 
 - <b>`theta`</b>:  Threshold parameter. 
 - <b>`M`</b> (float):  Margin parameter. 
 - <b>`alpha`</b> (float):  Promotion parameter. 
 - <b>`beta`</b> (float):  Demotion parameter. 
 - <b>`reset_after_drift`</b>:  A boolean indicating if the change detector will be reset after a drift was detected. baseline:  A string identifier of the baseline method. The baseline is the value that we substitute non-selected  features with. This is necessary, because most online learning models are not able to handle arbitrary  patterns of missing data. ref_sample:  A sample used to compute the baseline. If the constant baseline is used, one needs to provide a single  float value. 




---

<a href="https://github.com/haugjo/float/tree/main/float/feature_selection/efs.py#L107"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `EFS.reset`

```python
reset()
```

Resets the feature selector. 

---

<a href="https://github.com/haugjo/float/tree/main/float/feature_selection/efs.py#L72"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `EFS.weight_features`

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
