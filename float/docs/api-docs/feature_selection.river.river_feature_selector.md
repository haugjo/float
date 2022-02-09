<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/feature_selection/river/river_feature_selector.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `feature_selection.river.river_feature_selector`
River Feature Selection Model Wrapper. 

This module contains a wrapper class for river feature selection models. 

Copyright (C) 2022 Johannes Haug. 



---

<a href="https://github.com/haugjo/float/tree/main/float/feature_selection/river/river_feature_selector.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `RiverFeatureSelector`
Wrapper for river feature selection models. 

<a href="https://github.com/haugjo/float/tree/main/float/feature_selection/river/river_feature_selector.py#L23"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `RiverFeatureSelector.__init__`

```python
__init__(
    model: river.base.transformer.Transformer,
    feature_names: List[str],
    n_total_features: int,
    reset_after_drift: bool = False,
    baseline: str = 'constant',
    ref_sample: Union[float, numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]] = 0
)
```

Inits the wrapper. 



**Args:**
 
 - <b>`model`</b>:  The river feature selector object (one of SelectKBest, PoissonInclusion or VarianceThreshold). 
 - <b>`feature_names`</b>:  A list of all feature names. 
 - <b>`n_total_features`</b>:  The total number of features. 
 - <b>`reset_after_drift`</b>:  A boolean indicating if the change detector will be reset after a drift was detected. baseline:  A string identifier of the baseline method. The baseline is the value that we substitute non-selected  features with. This is necessary, because most online learning models are not able to handle arbitrary  patterns of missing data. ref_sample:  A sample used to compute the baseline. If the constant baseline is used, one needs to provide a single  float value. 




---

<a href="https://github.com/haugjo/float/tree/main/float/feature_selection/river/river_feature_selector.py#L125"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `RiverFeatureSelector.reset`

```python
reset()
```

Resets the feature selector. 

---

<a href="https://github.com/haugjo/float/tree/main/float/feature_selection/river/river_feature_selector.py#L77"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `RiverFeatureSelector.select_features`

```python
select_features(
    X: Union[numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]],
    rng: numpy.random._generator.Generator
) → Union[numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]]
```

Selects features with highest absolute weights. 

This overrides the corresponding parent class function. 



**Args:**
 
 - <b>`X`</b>:  Array/matrix of observations. 
 - <b>`rng`</b>:  A numpy random number generator object. 



**Returns:**
 ArrayLike:  The observation array/matrix where all non-selected features have been replaced by the baseline value. 

---

<a href="https://github.com/haugjo/float/tree/main/float/feature_selection/river/river_feature_selector.py#L55"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `RiverFeatureSelector.weight_features`

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
