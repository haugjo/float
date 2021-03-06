<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/evaluation/measures/missed_detection_rate.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `change_detection.evaluation.measures.missed_detection_rate`
Missed Detection Rate Measure. 

This function returns the rate of known concept drifts that the model has failed to detect. The measure was introduced by Bifet, Albert, et al. "CD-MOA: Change detection framework for massive online analysis." International Symposium on Intelligent Data Analysis. Springer, Berlin, Heidelberg, 2013. 

The missed detection rate is equal to 1 - detected_change_rate. 

Copyright (C) 2022 Johannes Haug. 


---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/evaluation/measures/missed_detection_rate.py#L34"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `missed_detection_rate`

```python
missed_detection_rate(
    evaluator: float.change_detection.evaluation.change_detection_evaluator.ChangeDetectionEvaluator,
    drifts: list,
    n_delay: int
) → float
```

Calculates the rate of missed known concept drifts. 

The missed detection rate is equal to 1 - the detected change rate. 



**Args:**
 
 - <b>`evaluator`</b>:  The ChangeDetectionEvaluator object. 
 - <b>`drifts`</b>:  List of time steps corresponding to detected concept drifts. 
 - <b>`n_delay`</b>:  The number of observations after a known concept drift, during which we count the detections made by the  model as true positives. 



**Returns:**
 
 - <b>`float`</b>:  The fraction of missed known concept drifts. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
