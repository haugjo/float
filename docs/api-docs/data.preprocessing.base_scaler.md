<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/data/preprocessing/base_scaler.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `data.preprocessing.base_scaler`
Base Scaler. 

This module encapsulates functionality to scale, i.e. normalize, streaming observations. The abstract BaseScaler should be used to implement custom scaling methods. A scaler object can be provided to the data loader object. 

Copyright (C) 2022 Johannes Haug. 



---

<a href="https://github.com/haugjo/float/tree/main/float/data/preprocessing/base_scaler.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BaseScaler`
Abstract Base Class for online data scaling. 



**Attributes:**
 
 - <b>`reset_after_drift`</b> (bool):  A boolean indicating if the scaler will be reset after a drift was detected. 

<a href="https://github.com/haugjo/float/tree/main/float/data/preprocessing/base_scaler.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `BaseScaler.__init__`

```python
__init__(reset_after_drift: bool)
```

Initializes the data scaler. 



**Args:**
 
 - <b>`reset_after_drift`</b>:  A boolean indicating if the scaler will be reset after a drift was detected. 




---

<a href="https://github.com/haugjo/float/tree/main/float/data/preprocessing/base_scaler.py#L26"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `BaseScaler.partial_fit`

```python
partial_fit(
    X: Union[numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]]
)
```

Updates the scaler. 



**Args:**
 
 - <b>`X`</b>:  Array/matrix of observations. 

---

<a href="https://github.com/haugjo/float/tree/main/float/data/preprocessing/base_scaler.py#L47"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `BaseScaler.reset`

```python
reset()
```

Resets the scaler. 

---

<a href="https://github.com/haugjo/float/tree/main/float/data/preprocessing/base_scaler.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `BaseScaler.transform`

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
