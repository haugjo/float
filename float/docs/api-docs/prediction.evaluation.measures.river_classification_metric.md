<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/prediction/evaluation/measures/river_classification_metric.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `prediction.evaluation.measures.river_classification_metric`
River Classification Metric Wrapper. 

This function is a wrapper for river classification metrics. This wrapper is required, as river metrics cannot process batches of observations out of the box. 

Copyright (C) 2022 Johannes Haug. 


---

<a href="https://github.com/haugjo/float/tree/main/float/prediction/evaluation/measures/river_classification_metric.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `river_classification_metric`

```python
river_classification_metric(
    y_true: Union[numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]],
    y_pred: Union[numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]],
    metric: river.metrics.base.ClassificationMetric,
    **kwargs
) → Any
```

Wrapper function for river classification metrics. 



**Args:**
 
 - <b>`y_true`</b>:  True target labels. 
 - <b>`y_pred`</b>:  Predicted target labels. 
 - <b>`metric`</b>:  The river classification metric. 

**kwargs:**
  A dictionary containing additional and specific keyword arguments, which are passed to the evaluation  functions. 



**Returns:**
 
 - <b>`Any`</b>:  The current value of the specified metric. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
