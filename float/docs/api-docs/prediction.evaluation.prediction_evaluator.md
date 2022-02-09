<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/prediction/evaluation/prediction_evaluator.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `prediction.evaluation.prediction_evaluator`
Predictive Model Evaluator. 

This module contains an evaluator class for online predictive models. 

Copyright (C) 2022 Johannes Haug. 



---

<a href="https://github.com/haugjo/float/tree/main/float/prediction/evaluation/prediction_evaluator.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `PredictionEvaluator`
Online prediction evaluator class. 



**Attributes:**
 
 - <b>`measure_funcs`</b> (List[Callable]):  List of evaluation measure functions. 
 - <b>`decay_rate`</b> (float |None):  If this parameter is not None, the measurements are additionally aggregated with the specific decay/fading  factor. 
 - <b>`window_size`</b> (int | None):  If this parameter is not None, the measurements are additionally aggregated in a sliding window. 
 - <b>`kwargs`</b> (dict):  A dictionary containing additional and specific keyword arguments, which are passed to the evaluation  functions. 
 - <b>`testing_comp_times`</b> (list):  List of computation times per testing iteration. 
 - <b>`training_comp_times`</b> (list):  List of computation times per training iteration. 
 - <b>`memory_changes`</b> (list):  Memory changes (in GB RAM) per training iteration of the online feature selection model. 
 - <b>`result`</b> (dict):  The raw and aggregated measurements of each evaluation measure function. 

<a href="https://github.com/haugjo/float/tree/main/float/prediction/evaluation/prediction_evaluator.py#L38"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `PredictionEvaluator.__init__`

```python
__init__(
    measure_funcs: List[Callable],
    decay_rate: Optional[float] = None,
    window_size: Optional[float] = None,
    **kwargs
)
```

Inits the prediction evaluation object. 



**Args:**
 
 - <b>`measure_funcs`</b>:  List of evaluation measure functions. 
 - <b>`decay_rate`</b>:  If this parameter is not None, the measurements are additionally aggregated with the specific  decay/fading factor. 
 - <b>`window_size`</b>:  If this parameter is not None, the measurements are additionally aggregated in a sliding window. 

**kwargs:**
  A dictionary containing additional and specific keyword arguments, which are passed to the evaluation  functions. 




---

<a href="https://github.com/haugjo/float/tree/main/float/prediction/evaluation/prediction_evaluator.py#L83"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `PredictionEvaluator.run`

```python
run(
    y_true: Union[numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]],
    y_pred: Union[numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]],
    X: Union[numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]],
    predictor: float.prediction.base_predictor.BasePredictor,
    rng: numpy.random._generator.Generator
)
```

Updates relevant statistics and computes the evaluation measures. 



**Args:**
 
 - <b>`y_true`</b>:  True target labels. 
 - <b>`y_pred`</b>:  Predicted target labels. 
 - <b>`X`</b>:  Array/matrix of observations. 
 - <b>`predictor`</b>:  Predictor object. 
 - <b>`rng`</b>:  A numpy random number generator object. 



**Raises:**
 
 - <b>`TypeError`</b>:  If the calculation of a measure runs an error. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
