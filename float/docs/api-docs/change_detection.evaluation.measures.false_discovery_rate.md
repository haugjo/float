<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/evaluation/measures/false_discovery_rate.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `change_detection.evaluation.measures.false_discovery_rate`
False Discovery Rate Measure. 

This function returns the false discovery rate, i.e. the fraction of false positives among all detected drifts. 

Copyright (C) 2022 Johannes Haug. 


---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/evaluation/measures/false_discovery_rate.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `false_discovery_rate`

```python
false_discovery_rate(
    evaluator: float.change_detection.evaluation.change_detection_evaluator.ChangeDetectionEvaluator,
    drifts: list,
    n_delay: int
) → float
```

Calculates the false discovery rate of detected drifts. 



**Args:**
 
 - <b>`evaluator`</b>:  The ChangeDetectionEvaluator object. 
 - <b>`drifts`</b>:  List of time steps corresponding to detected concept drifts. n_delay:  The number of observations after a known concept drift, during which we count the detections made by the  model as true positives. 



**Returns:**
 
 - <b>`float`</b>:  The false discovery rate of detected concept drifts. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
