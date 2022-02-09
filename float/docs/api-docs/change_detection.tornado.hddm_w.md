<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/hddm_w.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `change_detection.tornado.hddm_w`
Hoeffding's Bound based Drift Detection Method (W_test Scheme). 

The source code was adopted from https://github.com/alipsgh/tornado, please cite: The Tornado Framework By Ali Pesaranghader University of Ottawa, Ontario, Canada E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com 
--- Original Paper: Frías-Blanco, Isvani, et al. "Online and non-parametric drift detection methods based on Hoeffding’s bounds." Published in: IEEE Transactions on Knowledge and Data Engineering 27.3 (2015): 810-823. URL: http://ieeexplore.ieee.org/abstract/document/6871418/ 

Copyright (C) 2022 Johannes Haug. 



---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/hddm_w.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `HDDMW`
HDDMW change detector. 

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/hddm_w.py#L24"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `HDDMW.__init__`

```python
__init__(
    drift_confidence: float = 0.001,
    warning_confidence: float = 0.005,
    lambda_: float = 0.05,
    test_type: str = 'one-sided',
    reset_after_drift: bool = False
)
```

Inits the change detector. 



**Args:**
 
 - <b>`drift_confidence`</b>:  Todo (left unspecified by the Tornado library). 
 - <b>`warning_confidence`</b>:  Todo (left unspecified by the Tornado library). 
 - <b>`lambda_`</b>:  Todo (left unspecified by the Tornado library). 
 - <b>`test_type`</b>:  Todo (left unspecified by the Tornado library). 
 - <b>`reset_after_drift`</b>:  A boolean indicating if the change detector will be reset after a drift was detected. 




---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/hddm_w.py#L96"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `HDDMW.detect_change`

```python
detect_change() → bool
```

Detects global concept drift. 



**Returns:**
 
 - <b>`bool`</b>:  True, if a concept drift was detected, False otherwise. 

---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/hddm_w.py#L104"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `HDDMW.detect_partial_change`

```python
detect_partial_change() → Tuple[bool, list]
```

Detects partial concept drift. 



**Notes:**

> HDDMW does not detect partial change. 

---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/hddm_w.py#L112"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `HDDMW.detect_warning_zone`

```python
detect_warning_zone() → bool
```

Detects a warning zone. 



**Returns:**
 
 - <b>`bool`</b>:  True, if the change detector has detected a warning zone, False otherwise. 

---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/hddm_w.py#L59"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `HDDMW.partial_fit`

```python
partial_fit(pr_scores: List[bool])
```

Updates the change detector. 



**Args:**
  pr_scores:  A boolean vector indicating correct predictions. 'True' values indicate that the prediction by the  online learner was correct, otherwise the vector contains 'False'. 

---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/hddm_w.py#L55"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `HDDMW.reset`

```python
reset()
```

Resets the change detector. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
