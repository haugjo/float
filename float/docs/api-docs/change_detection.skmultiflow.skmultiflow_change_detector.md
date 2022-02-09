<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/skmultiflow/skmultiflow_change_detector.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `change_detection.skmultiflow.skmultiflow_change_detector`
Scikit-Multiflow Change Detection Model Wrapper. 

This module contains a wrapper class for [scikit-multiflow](https://scikit-multiflow.readthedocs.io/en/stable/) concept drift detection methods. 

Copyright (C) 2022 Johannes Haug. 



---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/skmultiflow/skmultiflow_change_detector.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SkmultiflowChangeDetector`
Wrapper class for scikit-multiflow change detection classes. 



**Attributes:**
 
 - <b>`detector`</b> (BaseDriftDetector):  The scikit-multiflow concept drift detector object. 

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/skmultiflow/skmultiflow_change_detector.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `SkmultiflowChangeDetector.__init__`

```python
__init__(
    detector: skmultiflow.drift_detection.base_drift_detector.BaseDriftDetector,
    reset_after_drift: bool = False
)
```

Inits the wrapper. 



**Args:**
 
 - <b>`detector`</b>:  The scikit-multiflow concept drift detector object. 
 - <b>`reset_after_drift`</b>:  A boolean indicating if the change detector will be reset after a drift was detected. 




---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/skmultiflow/skmultiflow_change_detector.py#L46"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `SkmultiflowChangeDetector.detect_change`

```python
detect_change() → bool
```

Detects global concept drift. 



**Returns:**
 
 - <b>`bool`</b>:  True, if a concept drift was detected, False otherwise. 

---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/skmultiflow/skmultiflow_change_detector.py#L54"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `SkmultiflowChangeDetector.detect_partial_change`

```python
detect_partial_change() → Tuple[bool, list]
```

Detects partial concept drift. 



**Notes:**

> Scikit-multiflow change detectors do not detect partial change. 

---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/skmultiflow/skmultiflow_change_detector.py#L62"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `SkmultiflowChangeDetector.detect_warning_zone`

```python
detect_warning_zone() → bool
```

Detects a warning zone. 



**Returns:**
 
 - <b>`bool`</b>:  True, if the change detector has detected a warning zone, False otherwise. 

---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/skmultiflow/skmultiflow_change_detector.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `SkmultiflowChangeDetector.partial_fit`

```python
partial_fit(pr_scores: List[bool])
```

Updates the change detector. 



**Args:**

- <b>`pr_scores`</b>: A boolean vector indicating correct predictions. 'True' values indicate that the prediction by the  online learner was correct, otherwise the vector contains 'False'. 

---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/skmultiflow/skmultiflow_change_detector.py#L31"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `SkmultiflowChangeDetector.reset`

```python
reset()
```

Resets the change detector. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
