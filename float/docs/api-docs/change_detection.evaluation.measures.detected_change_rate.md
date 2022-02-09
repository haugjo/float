<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/evaluation/measures/detected_change_rate.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `change_detection.evaluation.measures.detected_change_rate`
Detected Change Rate Measure. 

This function returns the fraction of correctly detected known drifts. The detected change rate measure is sometimes also called recall or false positive rate. 

Copyright (C) 2022 Johannes Haug. 


---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/evaluation/measures/detected_change_rate.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `detected_change_rate`

```python
detected_change_rate(
    evaluator: float.change_detection.evaluation.change_detection_evaluator.ChangeDetectionEvaluator,
    drifts: list,
    n_delay: int
) → float
```

Calculates the rate of correctly detected known concept drifts. 



**Args:**
 
 - <b>`evaluator`</b>:  The ChangeDetectionEvaluator object. 
 - <b>`drifts`</b>:  List of time steps corresponding to detected concept drifts. 
 - <b>`n_delay`</b>:  The number of observations after a known concept drift, during which we count the detections made by the  model as true positives. 



**Returns:**
 
 - <b>`float`</b>:  The rate of correctly detected known concept drifts 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
