<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/prediction/river/river_classifier.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `prediction.river.river_classifier`
River Predictive Model Wrapper. 

This module contains a wrapper class for [river](https://riverml.xyz/latest/) predictive models. 

Copyright (C) 2022 Johannes Haug. 



---

<a href="https://github.com/haugjo/float/tree/main/float/prediction/river/river_classifier.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `RiverClassifier`
Wrapper for river predictive models. 



**Attributes:**
 
 - <b>`model`</b> (ClassifierMixin):  The river predictor object. 
 - <b>`feature_names`</b> (List[str]):  A list of all feature names. 

<a href="https://github.com/haugjo/float/tree/main/float/prediction/river/river_classifier.py#L23"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `RiverClassifier.__init__`

```python
__init__(
    model: river.base.classifier.Classifier,
    feature_names: List[str],
    reset_after_drift: bool = False
)
```

Inits the wrapper. 



**Args:**
 
 - <b>`model`</b>:  The river predictor object. 
 - <b>`feature_names`</b>:  A list of all feature names. 
 - <b>`reset_after_drift`</b>:  A boolean indicating if the predictor will be reset after a drift was detected. 




---

<a href="https://github.com/haugjo/float/tree/main/float/prediction/river/river_classifier.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `RiverClassifier.partial_fit`

```python
partial_fit(
    X: Union[numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]],
    y: Union[numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]],
    sample_weight: Optional[numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]] = None
)
```

Updates the predictor. 



**Args:**
 
 - <b>`X`</b>:  Array/matrix of observations. 
 - <b>`y`</b>:  Array of corresponding labels. 
 - <b>`sample_weight`</b>:  Weights per sample. Not used by float at the moment, i.e., all observations in x receive equal weight in a pipeline run. 

---

<a href="https://github.com/haugjo/float/tree/main/float/prediction/river/river_classifier.py#L61"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `RiverClassifier.predict`

```python
predict(
    X: Union[numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]]
) → Union[numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]]
```

Predicts the target values. 



**Args:**
 
 - <b>`X`</b>:  Array/matrix of observations. 



**Returns:**
 
 - <b>`ArrayLike`</b>:  Predicted labels for all observations. 

---

<a href="https://github.com/haugjo/float/tree/main/float/prediction/river/river_classifier.py#L77"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `RiverClassifier.predict_proba`

```python
predict_proba(
    X: Union[numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]]
) → Union[numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]]
```

Predicts the probability of target values. 



**Args:**
 
 - <b>`X`</b>:  Array/matrix of observations. 



**Returns:**
 
 - <b>`ArrayLike`</b>:  Predicted probability per class label for all observations. 

---

<a href="https://github.com/haugjo/float/tree/main/float/prediction/river/river_classifier.py#L93"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `RiverClassifier.reset`

```python
reset()
```

Resets the predictor. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
