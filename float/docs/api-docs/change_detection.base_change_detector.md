<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/base_change_detector.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `change_detection.base_change_detector`
Base Change Detection Module. 

This module encapsulates functionality for global and partial (i.e. feature-wise) concept drift detection. The abstract BaseChangeDetector class should be used as super class for all concept drift detection methods. 

Copyright (C) 2022 Johannes Haug. 



---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/base_change_detector.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BaseChangeDetector`
Abstract base class for change detection models. 



**Attributes:**
 
 - <b>`reset_after_drift`</b> (bool):  A boolean indicating if the change detector will be reset after a drift was detected. error_based (bool):  A boolean indicating if the change detector partial_fit function requires error measurements (i.e., a  boolean vector indicating correct predictions) as input from the pipeline. This is true for most change  detectors. If the attribute is False, the partial_fit function will receive raw input observations and  targets from the pipeline (e.g., required by ERICS). 
 - <b>`drifts`</b> (list):  A list of time steps corresponding to detected concept drifts. partial_drifts (List[tuple]):  A list of time steps and features corresponding to detected partial concept drifts. A partial drift is a  concept drift that is restricted to one or multiple (but not all) input features. Some change detectors are  able to detect partial concept drift. This attribute is a list of tuples of the form (time step,  [features under change]). warnings (list):  A list of time steps corresponding to warnings. Some change detectors are able to issue warnings before  an actual drift alert. 

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/base_change_detector.py#L32"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `BaseChangeDetector.__init__`

```python
__init__(reset_after_drift: bool, error_based: bool)
```

Inits the change detector. 



**Args:**
 
 - <b>`reset_after_drift`</b>:  A boolean indicating if the change detector will be reset after a drift was detected. error_based:  A boolean indicating if the change detector partial_fit function requires error measurements (i.e., a  boolean vector indicating correct predictions) as input from the pipeline. This is true for most change  detectors. If the attribute is False, the partial_fit function will receive raw input observations and  targets from the pipeline (e.g., required by ERICS). 




---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/base_change_detector.py#L59"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `BaseChangeDetector.detect_change`

```python
detect_change() → bool
```

Detects global concept drift. 



**Returns:**
 
 - <b>`bool`</b>:  True, if a concept drift was detected, False otherwise. 

---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/base_change_detector.py#L68"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `BaseChangeDetector.detect_partial_change`

```python
detect_partial_change() → Tuple[bool, list]
```

Detects partial concept drift. 



**Returns:**
 
 - <b>`bool`</b>:  True, if at least one partial concept drift was detected, False otherwise. 
 - <b>`list`</b>:  Indices (i.e. relative positions in the feature vector) of input features with detected partial drift. 

---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/base_change_detector.py#L78"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `BaseChangeDetector.detect_warning_zone`

```python
detect_warning_zone() → bool
```

Detects a warning zone. 

Some change detectors issue warnings before the actual drift alert. 



**Returns:**
 
 - <b>`bool`</b>:  True, if the change detector has detected a warning zone, False otherwise. 

---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/base_change_detector.py#L54"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `BaseChangeDetector.partial_fit`

```python
partial_fit(*args, **kwargs)
```

Updates the change detector. 

---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/base_change_detector.py#L49"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `BaseChangeDetector.reset`

```python
reset()
```

Resets the change detector. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
