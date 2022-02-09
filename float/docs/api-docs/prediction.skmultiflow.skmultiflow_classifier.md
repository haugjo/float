<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/prediction/skmultiflow/skmultiflow_classifier.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `prediction.skmultiflow.skmultiflow_classifier`
Scikit-Multiflow Predictive Model Wrapper. 

This module contains a wrapper class for [scikit-multiflow](https://scikit-multiflow.readthedocs.io/en/stable/) predictive models. 

Copyright (C) 2022 Johannes Haug. 



---

<a href="https://github.com/haugjo/float/tree/main/float/prediction/skmultiflow/skmultiflow_classifier.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SkmultiflowClassifier`
Wrapper for scikit-multiflow predictive models. 



**Attributes:**
 
 - <b>`model`</b> (ClassifierMixin):  The scikit-multiflow predictor object. 
 - <b>`classes`</b> (list):  A list of all unique classes. 

<a href="https://github.com/haugjo/float/tree/main/float/prediction/skmultiflow/skmultiflow_classifier.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `SkmultiflowClassifier.__init__`

```python
__init__(
    model: skmultiflow.core.base.ClassifierMixin,
    classes: list,
    reset_after_drift: bool = False
)
```

Inits the wrapper. 



**Args:**
 
 - <b>`model`</b>:  The scikit-multiflow predictor object. 
 - <b>`classes`</b>:  A list of all unique classes. 
 - <b>`reset_after_drift`</b>:  A boolean indicating if the predictor will be reset after a drift was detected. 




---

<a href="https://github.com/haugjo/float/tree/main/float/prediction/skmultiflow/skmultiflow_classifier.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `SkmultiflowClassifier.partial_fit`

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

<a href="https://github.com/haugjo/float/tree/main/float/prediction/skmultiflow/skmultiflow_classifier.py#L43"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `SkmultiflowClassifier.predict`

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

<a href="https://github.com/haugjo/float/tree/main/float/prediction/skmultiflow/skmultiflow_classifier.py#L54"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `SkmultiflowClassifier.predict_proba`

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

<a href="https://github.com/haugjo/float/tree/main/float/prediction/skmultiflow/skmultiflow_classifier.py#L65"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `SkmultiflowClassifier.reset`

```python
reset()
```

Resets the predictor. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
