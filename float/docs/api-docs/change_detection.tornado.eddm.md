<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/eddm.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `change_detection.tornado.eddm`
Early Drift Detection Method. 

The source code was adopted from https://github.com/alipsgh/tornado, please cite: The Tornado Framework By Ali Pesaranghader University of Ottawa, Ontario, Canada E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com 
--- Original Paper: Baena-García, Manuel, et al. "Early drift detection method." (2006). URL: http://www.cs.upc.edu/~abifet/EDDM.pdf 

Copyright (C) 2022 Johannes Haug. 



---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/eddm.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `EDDM`
EDDM change detector. 

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/eddm.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `EDDM.__init__`

```python
__init__(reset_after_drift: bool = False)
```

Inits the change detector. 



**Args:**
 
 - <b>`reset_after_drift`</b>:  A boolean indicating if the change detector will be reset after a drift was detected. 




---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/eddm.py#L92"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `EDDM.detect_change`

```python
detect_change() → bool
```

Detects global concept drift. 



**Returns:**
 
 - <b>`bool`</b>:  True, if a concept drift was detected, False otherwise. 

---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/eddm.py#L100"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `EDDM.detect_partial_change`

```python
detect_partial_change() → Tuple[bool, list]
```

Detects partial concept drift. 



**Notes:**

> EDDM does not detect partial change. 

---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/eddm.py#L108"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `EDDM.detect_warning_zone`

```python
detect_warning_zone() → bool
```

Detects a warning zone. 



**Returns:**
 
 - <b>`bool`</b>:  True, if the change detector has detected a warning zone, False otherwise. 

---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/eddm.py#L54"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `EDDM.partial_fit`

```python
partial_fit(pr_scores: List[bool])
```

Updates the change detector. 



**Args:**
  pr_scores:  A boolean vector indicating correct predictions. 'True' values indicate that the prediction by the  online learner was correct, otherwise the vector contains 'False'. 

---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/eddm.py#L44"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `EDDM.reset`

```python
reset()
```

Resets the change detector. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
