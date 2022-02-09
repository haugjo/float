<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/data/preprocessing/sklearn_scaler.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `data.preprocessing.sklearn_scaler`
Sklearn Scaling Function Wrapper. 

This module contains a wrapper for the scikit-learn scaling functions. 

Copyright (C) 2022 Johannes Haug. 



---

<a href="https://github.com/haugjo/float/tree/main/float/data/preprocessing/sklearn_scaler.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SklearnScaler`
Wrapper for sklearn scaler functions. 



**Attributes:**
 
 - <b>`scaler_obj`</b> (Any):  A scikit-learn scaler object (e.g. MinMaxScaler) 

<a href="https://github.com/haugjo/float/tree/main/float/data/preprocessing/sklearn_scaler.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `SklearnScaler.__init__`

```python
__init__(scaler_obj: Any, reset_after_drift: bool = False)
```

Inits the sklearn scaler. 



**Args:**
 
 - <b>`scaler_obj`</b>:  A scikit-learn scaler object (e.g. MinMaxScaler) 
 - <b>`reset_after_drift`</b>:  A boolean indicating if the scaler will be reset after a drift was detected. 




---

<a href="https://github.com/haugjo/float/tree/main/float/data/preprocessing/sklearn_scaler.py#L36"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `SklearnScaler.partial_fit`

```python
partial_fit(
    X: Union[numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]]
)
```

Updates the scaler. 



**Args:**
 
 - <b>`X`</b>:  Array/matrix of observations. 

---

<a href="https://github.com/haugjo/float/tree/main/float/data/preprocessing/sklearn_scaler.py#L59"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `SklearnScaler.reset`

```python
reset()
```

Resets the scaler. 

We automatically re-fit the scaler upon the next call to partial_fit. 

---

<a href="https://github.com/haugjo/float/tree/main/float/data/preprocessing/sklearn_scaler.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `SklearnScaler.transform`

```python
transform(
    X: Union[numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]]
) → Union[numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]]
```

Scales the given observations. 



**Args:**
 
 - <b>`X`</b>:  Array/matrix of observations. 



**Returns:**
 
 - <b>`ArrayLike`</b>:  The scaled observations. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
