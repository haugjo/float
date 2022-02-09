<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/feature_selection/base_feature_selector.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `feature_selection.base_feature_selector`
Base Online Feature Selection Module. 

This module encapsulates functionality for online feature weighting and selection. The abstract BaseFeatureSelector class should be used as a super class for all online feature selection methods. 

Copyright (C) 2022 Johannes Haug. 



---

<a href="https://github.com/haugjo/float/tree/main/float/feature_selection/base_feature_selector.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BaseFeatureSelector`
Abstract base class for online feature selection methods. 



**Attributes:**
 
 - <b>`n_total_features`</b> (int):  The total number of features. 
 - <b>`n_selected_features`</b> (int):  The number of selected features. supports_multi_class (bool):  True if the feature selection model supports multi-class classification, False otherwise. 
 - <b>`reset_after_drift`</b> (bool):  A boolean indicating if the change detector will be reset after a drift was detected. baseline (str):  A string identifier of the baseline method. The baseline is the value that we substitute non-selected  features with. This is necessary, because most online learning models are not able to handle arbitrary  patterns of missing data. ref_sample (ArrayLike | float):  A sample used to compute the baseline. If the constant baseline is used, one needs to provide a single  float value. 
 - <b>`weights`</b> (ArrayLike):  The current (raw) feature weights. 
 - <b>`selected_features`</b> (ArrayLike):  The indices of all currently selected features. 
 - <b>`weights_history`</b> (List[list]):  A list of all absolute feature weight vectors obtained over time. 
 - <b>`selected_features_history`</b> (List[list]):  A list of all selected feature vectors obtained over time. 

<a href="https://github.com/haugjo/float/tree/main/float/feature_selection/base_feature_selector.py#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `BaseFeatureSelector.__init__`

```python
__init__(
    n_total_features: int,
    n_selected_features: int,
    supports_multi_class: bool,
    reset_after_drift: bool,
    baseline: str,
    ref_sample: Union[float, numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]]
)
```

Inits the feature selector. 



**Args:**
 
 - <b>`n_total_features`</b>:  The total number of features. 
 - <b>`n_selected_features`</b>:  The number of selected features. supports_multi_class:  True if the feature selection model supports multi-class classification, False otherwise. 
 - <b>`reset_after_drift`</b>:  A boolean indicating if the change detector will be reset after a drift was detected. baseline:  A string identifier of the baseline method. The baseline is the value that we substitute non-selected  features with. This is necessary, because most online learning models are not able to handle arbitrary  patterns of missing data. ref_sample:  A sample used to compute the baseline. If the constant baseline is used, one needs to provide a single  float value. 




---

<a href="https://github.com/haugjo/float/tree/main/float/feature_selection/base_feature_selector.py#L88"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `BaseFeatureSelector.reset`

```python
reset()
```

Resets the feature selector. 

---

<a href="https://github.com/haugjo/float/tree/main/float/feature_selection/base_feature_selector.py#L93"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `BaseFeatureSelector.select_features`

```python
select_features(
    X: Union[numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]],
    rng: numpy.random._generator.Generator
) → Union[numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]]
```

Selects features with highest absolute weights. 



**Args:**
 
 - <b>`X`</b>:  Array/matrix of observations. 
 - <b>`rng`</b>:  A numpy random number generator object. 



**Returns:**
 ArrayLike:  The observation array/matrix where all non-selected features have been replaced by the baseline value. 

---

<a href="https://github.com/haugjo/float/tree/main/float/feature_selection/base_feature_selector.py#L78"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `BaseFeatureSelector.weight_features`

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
