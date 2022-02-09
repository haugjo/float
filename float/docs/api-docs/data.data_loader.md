<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/data/data_loader.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `data.data_loader`
Data Loader. 

This module encapsulates functionality to load and preprocess input data. The data loader class uses the scikit-multiflow Stream class to simulate streaming data. 

Copyright (C) 2022 Johannes Haug. 



---

<a href="https://github.com/haugjo/float/tree/main/float/data/data_loader.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DataLoader`
Data Loader Class. 

The data loader class is responsible to sample and pre-process (i.e. normalize) input data, thereby simulating a data stream. The data loader uses a skmultiflow Stream object to generate or load streaming data. 



**Attributes:**
 
 - <b>`path`</b> (str | None):  The path to a .csv file containing the training data set. 
 - <b>`stream`</b> (Stream | None):  A scikit-multiflow data stream object. 
 - <b>`target_col`</b> (int):  The index of the target column in the training data. 
 - <b>`scaler`</b> (BaseScaler | None):  A scaler object used to normalize/standardize sampled instances. 

<a href="https://github.com/haugjo/float/tree/main/float/data/data_loader.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DataLoader.__init__`

```python
__init__(
    path: Optional[str] = None,
    stream: Optional[skmultiflow.data.base_stream.Stream] = None,
    target_col: int = -1,
    scaler: Optional[float.data.preprocessing.base_scaler.BaseScaler] = None
)
```

Inits the data loader. 

The data loader init function must receive either one of the following inputs:
1.) the path to a .csv file (+ a target index), which is then mapped to a skmultiflow FileStream object.
2.) a valid scikit multiflow Stream object. 



**Args:**
 
 - <b>`path`</b>:  The path to a .csv file containing the training data set. 
 - <b>`stream`</b>:  A scikit-multiflow data stream object. 
 - <b>`target_col`</b>:  The index of the target column in the training data. 
 - <b>`scaler`</b>:  A scaler object used to normalize/standardize sampled instances. 




---

<a href="https://github.com/haugjo/float/tree/main/float/data/data_loader.py#L53"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DataLoader.get_data`

```python
get_data(
    n_batch: int
) → Tuple[Union[numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]], Union[numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]]]
```

Loads a batch from the stream object. 



**Args:**
 
 - <b>`n_batch`</b>:  Number of samples to load from the data stream object. 



**Returns:**
 
 - <b>`Tuple[ArrayLike, ArrayLike]`</b>:  The sampled observations and corresponding targets. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
