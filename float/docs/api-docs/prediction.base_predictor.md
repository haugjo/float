<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/prediction/base_predictor.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `prediction.base_predictor`
Base Online Predictor Module. 

This module encapsulates functionality for online predictive modelling. The abstract BasePredictor class should be used as super class for all online predictive models. 

Copyright (C) 2022 Johannes Haug. 



---

<a href="https://github.com/haugjo/float/tree/main/float/prediction/base_predictor.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BasePredictor`
Abstract base class for online predictive models. 



**Attributes:**
 
 - <b>`reset_after_drift`</b> (bool):  A boolean indicating if the predictor will be reset after a drift was detected. 
 - <b>`has_been_trained`</b> (bool):  A boolean indicating if the predictor has been trained at least once. 

<a href="https://github.com/haugjo/float/tree/main/float/prediction/base_predictor.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `BasePredictor.__init__`

```python
__init__(reset_after_drift: bool)
```

Inits the predictor. 



**Args:**
 
 - <b>`reset_after_drift`</b>:  A boolean indicating if the predictor will be reset after a drift was detected. 




---

<a href="https://github.com/haugjo/float/tree/main/float/prediction/base_predictor.py#L30"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `BasePredictor.partial_fit`

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
 - <b>`sample_weight`</b>:  Weights per sample. If no weights are provided, we weigh observations uniformly. 

---

<a href="https://github.com/haugjo/float/tree/main/float/prediction/base_predictor.py#L41"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `BasePredictor.predict`

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

<a href="https://github.com/haugjo/float/tree/main/float/prediction/base_predictor.py#L53"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `BasePredictor.predict_proba`

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

<a href="https://github.com/haugjo/float/tree/main/float/prediction/base_predictor.py#L65"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `BasePredictor.reset`

```python
reset()
```

Resets the predictor. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
