<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/evaluation/change_detection_evaluator.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `change_detection.evaluation.change_detection_evaluator`
Change Detection Evaluator. 

This module contains an evaluator class for active change (i.e. concept drift) detection methods. 

Copyright (C) 2022 Johannes Haug. 



---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/evaluation/change_detection_evaluator.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ChangeDetectionEvaluator`
Change detection evaluation class. 

This class is required to compute the performance measures and store the corresponding results in the evaluation of the change detection method. 



**Attributes:**
 
 - <b>`measure_funcs`</b> (List[Callable]):  A list of evaluation measure functions. 
 - <b>`known_drifts`</b> (List[int] | List[tuple]):  The positions in the dataset (indices) corresponding to known concept drifts. 
 - <b>`batch_size`</b> (int):  The number of observations processed per iteration/time step. 
 - <b>`n_total`</b> (int):  The total number of observations. 
 - <b>`n_delay`</b> (int | list):  The number of observations after a known concept drift, during which we count the detections made by  the model as true positives. If the argument is a list, the evaluator computes results for each delay specified in the list. 
 - <b>`n_init_tolerance`</b> (int):  The number of observations reserved for the initial training. We do not consider these observations in the  evaluation. 
 - <b>`comp_times`</b> (list):  Computation times for updating the change detector per time step. 
 - <b>`memory_changes`</b> (list):  Memory changes (in GB RAM) per training iteration of the change detector. 
 - <b>`result`</b> (dict):  Results (i.e. calculated measurements, mean, and variance) for each evaluation measure function. 

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/evaluation/change_detection_evaluator.py#L36"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ChangeDetectionEvaluator.__init__`

```python
__init__(
    measure_funcs: List[Callable],
    known_drifts: Union[List[int], List[tuple]],
    batch_size: int,
    n_total: int,
    n_delay: Union[int, list] = 100,
    n_init_tolerance: int = 100
)
```

Initializes the change detection evaluation object. 



**Args:**
 
 - <b>`measure_funcs`</b>:  A list of evaluation measure functions. 
 - <b>`known_drifts`</b>:  The positions in the dataset (indices) corresponding to known concept drifts. 
 - <b>`batch_size`</b>:  The number of observations processed per iteration/time step. 
 - <b>`n_total`</b>:  The total number of observations. 
 - <b>`n_delay`</b>:  The number of observations after a known concept drift, during which we count the detections made by  the model as true positives. If the argument is a list, the evaluator computes results for each delay  specified in the list. 
 - <b>`n_init_tolerance`</b>:  The number of observations reserved for the initial training. We do not consider these observations in  the evaluation. 




---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/evaluation/change_detection_evaluator.py#L72"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ChangeDetectionEvaluator.run`

```python
run(drifts: List)
```

Computes the evaluation measures. 

Other than the PredictionEvaluator and FeatureSelectionEvaluator, the ChangeDetectionEvaluator is only run once at the end of the evaluation. 



**Args:**
 
 - <b>`drifts`</b>:  List of time steps corresponding to detected concept drifts. 



**Raises:**
 
 - <b>`TypeError`</b>:  Error while executing the provided evaluation measure functions. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
