<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/evaluation/measures/time_between_false_alarms.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `change_detection.evaluation.measures.time_between_false_alarms`
Time Between False Alarms Measure. 

This function returns the mean time between false alarms as introduced in: Bifet, Albert, et al. "CD-MOA: Change detection framework for massive online analysis." International Symposium on Intelligent Data Analysis. Springer, Berlin, Heidelberg, 2013. 

Copyright (C) 2022 Johannes Haug. 


---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/evaluation/measures/time_between_false_alarms.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `time_between_false_alarms`

```python
time_between_false_alarms(
    evaluator: float.change_detection.evaluation.change_detection_evaluator.ChangeDetectionEvaluator,
    drifts: list,
    n_delay: int
) → float
```

Calculates the mean time between false alarms. 



**Args:**
 
 - <b>`evaluator`</b>:  The ChangeDetectionEvaluator object. 
 - <b>`drifts`</b>:  List of time steps corresponding to detected concept drifts. 
 - <b>`n_delay`</b>:  The number of observations after a known concept drift, during which we count the detections made by the  model as true positives. 



**Returns:**
 
 - <b>`float`</b>:  The mean time between false alarms in number of observations. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
