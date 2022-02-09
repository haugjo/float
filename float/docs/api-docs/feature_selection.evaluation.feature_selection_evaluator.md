<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/feature_selection/evaluation/feature_selection_evaluator.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `feature_selection.evaluation.feature_selection_evaluator`
Evaluation Module for Online Feature Selection Methods. 

This module contains an evaluator class for online feature selection methods. 

Copyright (C) 2022 Johannes Haug. 



---

<a href="https://github.com/haugjo/float/tree/main/float/feature_selection/evaluation/feature_selection_evaluator.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FeatureSelectionEvaluator`
Online feature selection evaluator class. 



**Attributes:**
 
 - <b>`measure_funcs`</b> (List[Callable]):  List of evaluation measure functions. decay_rate (float |None):  If this parameter is not None, the measurements are additionally aggregated with the specific decay/fading  factor. window_size (int | None):  If this parameter is not None, the measurements are additionally aggregated in a sliding window. 
 - <b>`comp_times`</b> (list):  List of computation times per iteration of feature weighting and selection. memory_changes (list):  Memory changes (in GB RAM) per training iteration of the online feature selection model. 
 - <b>`result`</b> (dict):  The raw and aggregated measurements of each evaluation measure function. 

<a href="https://github.com/haugjo/float/tree/main/float/feature_selection/evaluation/feature_selection_evaluator.py#L27"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `FeatureSelectionEvaluator.__init__`

```python
__init__(
    measure_funcs: List[Callable],
    decay_rate: Optional[float] = None,
    window_size: Optional[int] = None
)
```

Inits the online feature selection evaluation object. 



**Args:**
 
 - <b>`measure_funcs`</b>:  List of evaluation measure functions. decay_rate:  If this parameter is not None, the measurements are additionally aggregated with the specific  decay/fading factor. window_size:  If this parameter is not None, the measurements are additionally aggregated in a sliding window. 




---

<a href="https://github.com/haugjo/float/tree/main/float/feature_selection/evaluation/feature_selection_evaluator.py#L62"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `FeatureSelectionEvaluator.run`

```python
run(selected_features_history: List[list], n_total_features: int)
```

Updates relevant statistics and computes the evaluation measures. 



**Args:**
 
 - <b>`selected_features_history`</b>:  A list of all selected feature vectors obtained over time. 
 - <b>`n_total_features`</b>:  The total number of features. 



**Raises:**
 
 - <b>`TypeError`</b>:  If the calculation of a measure runs an error. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
