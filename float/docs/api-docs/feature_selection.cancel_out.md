<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/feature_selection/cancel_out.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `feature_selection.cancel_out`
CancelOut Feature Selection Method. 

This module contains an adaptation of the CancelOut feature selection method for data streams. CancelOut was introduced as a feature selection layer for neural networks by: BORISOV, Vadim; HAUG, Johannes; KASNECI, Gjergji. Cancelout: A layer for feature selection in deep neural networks. In: International Conference on Artificial Neural Networks. Springer, Cham, 2019. S. 72-83. 

Copyright (C) 2022 Johannes Haug. 



---

<a href="https://github.com/haugjo/float/tree/main/float/feature_selection/cancel_out.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CancelOutFeatureSelector`
CancelOut feature selector. 

<a href="https://github.com/haugjo/float/tree/main/float/feature_selection/cancel_out.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `CancelOutFeatureSelector.__init__`

```python
__init__(
    n_total_features: int,
    n_selected_features: int,
    reset_after_drift: bool = False,
    baseline: str = 'constant',
    ref_sample: Union[float, numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]] = 0
)
```

Inits the feature selector. 



**Args:**
 
 - <b>`n_total_features`</b>:  The total number of features. 
 - <b>`n_selected_features`</b>:  The number of selected features. 
 - <b>`reset_after_drift`</b>:  A boolean indicating if the change detector will be reset after a drift was detected. baseline:  A string identifier of the baseline method. The baseline is the value that we substitute non-selected  features with. This is necessary, because most online learning models are not able to handle arbitrary  patterns of missing data. ref_sample:  A sample used to compute the baseline. If the constant baseline is used, one needs to provide a single  float value. 




---

<a href="https://github.com/haugjo/float/tree/main/float/feature_selection/cancel_out.py#L59"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `CancelOutFeatureSelector.reset`

```python
reset()
```

Resets the feature selector. 

CancelOut does not need to be reset, since the DNN is trained anew at every training iteration. 

---

<a href="https://github.com/haugjo/float/tree/main/float/feature_selection/cancel_out.py#L49"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `CancelOutFeatureSelector.weight_features`

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

<a href="https://github.com/haugjo/float/tree/main/float/feature_selection/cancel_out.py#L117"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CancelOutNeuralNet`
A neural network with CancelOut layer. 

<a href="https://github.com/haugjo/float/tree/main/float/feature_selection/cancel_out.py#L119"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `CancelOutNeuralNet.__init__`

```python
__init__(input_size, hidden_size, num_classes)
```








---

<a href="https://github.com/haugjo/float/tree/main/float/feature_selection/cancel_out.py#L127"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `CancelOutNeuralNet.forward`

```python
forward(x)
```






---

<a href="https://github.com/haugjo/float/tree/main/float/feature_selection/cancel_out.py#L137"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CancelOutDataLoader`
CancelOut dataset loader for the neural network. 

<a href="https://github.com/haugjo/float/tree/main/float/feature_selection/cancel_out.py#L139"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `CancelOutDataLoader.__init__`

```python
__init__(x, y)
```









---

<a href="https://github.com/haugjo/float/tree/main/float/feature_selection/cancel_out.py#L150"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CancelOut`
CancelOut network layer. 

<a href="https://github.com/haugjo/float/tree/main/float/feature_selection/cancel_out.py#L152"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `CancelOut.__init__`

```python
__init__(x)
```

Initializes the network layer 



**Args:**
 
 - <b>`X`</b>:  an input data (vector, matrix, tensor) 




---

<a href="https://github.com/haugjo/float/tree/main/float/feature_selection/cancel_out.py#L161"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `CancelOut.forward`

```python
forward(x)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
