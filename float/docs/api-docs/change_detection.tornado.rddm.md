<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/rddm.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `change_detection.tornado.rddm`
Reactive Drift Detection Method. 

The source code was adopted from https://github.com/alipsgh/tornado, please cite: The Tornado Framework By Ali Pesaranghader University of Ottawa, Ontario, Canada E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com 
--- Original Paper: Barros, Roberto, et al. "RDDM: Reactive drift detection method." Published in: Expert Systems with Applications. Elsevier, 2017. URL: https://www.sciencedirect.com/science/article/pii/S0957417417305614 

Copyright (C) 2022 Johannes Haug. 



---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/rddm.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `RDDM`
RDDM change detector. 

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/rddm.py#L24"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `RDDM.__init__`

```python
__init__(
    min_instance: int = 129,
    warning_level: float = 1.773,
    drift_level: float = 2.258,
    max_size_concept: int = 40000,
    min_size_stable_concept: int = 7000,
    warn_limit: int = 1400,
    reset_after_drift: bool = False
)
```

Inits the change detector. 



**Args:**
 
 - <b>`min_instance`</b>:  Todo (left unspecified by the Tornado library). 
 - <b>`warning_level`</b>:  Todo (left unspecified by the Tornado library). 
 - <b>`drift_level`</b>:  Todo (left unspecified by the Tornado library). 
 - <b>`max_size_concept`</b>:  Todo (left unspecified by the Tornado library). 
 - <b>`min_size_stable_concept`</b>:  Todo (left unspecified by the Tornado library). 
 - <b>`warn_limit`</b>:  Todo (left unspecified by the Tornado library). 
 - <b>`reset_after_drift`</b>:  A boolean indicating if the change detector will be reset after a drift was detected. 




---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/rddm.py#L174"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `RDDM.detect_change`

```python
detect_change() → bool
```

Detects global concept drift. 



**Returns:**
 
 - <b>`bool`</b>:  True, if a concept drift was detected, False otherwise. 

---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/rddm.py#L182"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `RDDM.detect_partial_change`

```python
detect_partial_change() → Tuple[bool, list]
```

Detects partial concept drift. 



**Notes:**

> RDDM does not detect partial change. 

---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/rddm.py#L190"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `RDDM.detect_warning_zone`

```python
detect_warning_zone() → bool
```

Detects a warning zone. 



**Returns:**
 
 - <b>`bool`</b>:  True, if the change detector has detected a warning zone, False otherwise. 

---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/rddm.py#L82"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `RDDM.partial_fit`

```python
partial_fit(pr_scores: List[bool])
```

Updates the change detector. 



**Args:**
  pr_scores:  A boolean vector indicating correct predictions. 'True' values indicate that the prediction by the  online learner was correct, otherwise the vector contains 'False'. 

---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/rddm.py#L73"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `RDDM.reset`

```python
reset()
```

Resets the change detector. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
