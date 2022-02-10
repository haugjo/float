<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/evaluation/measures/detection_delay.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `change_detection.evaluation.measures.detection_delay`
Detection Delay Measure. 

This function returns the average delay in number of observations between the beginning of a known concept drift and the first detected concept drift. 

Copyright (C) 2022 Johannes Haug. 


---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/evaluation/measures/detection_delay.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `detection_delay`

```python
detection_delay(
    evaluator: float.change_detection.evaluation.change_detection_evaluator.ChangeDetectionEvaluator,
    drifts: list,
    n_delay: Optional[int] = None
) → float
```

Calculates the average delay before detecting a concept drift. 



**Args:**
 
 - <b>`evaluator`</b>:  The ChangeDetectionEvaluator object. 
 - <b>`drifts`</b>:  List of time steps corresponding to detected concept drifts. 
 - <b>`n_delay`</b>:  This attribute is only included for consistency purposes. It is not relevant for this measure. 



**Returns:**
 
 - <b>`float`</b>:  The average delay in number of observations between a known drift and the first detected drift. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
