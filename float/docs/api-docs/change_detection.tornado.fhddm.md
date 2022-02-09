<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/fhddm.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `change_detection.tornado.fhddm`
Fast Hoeffding Drift Detection Method. 

The source code was adopted from https://github.com/alipsgh/tornado, please cite: The Tornado Framework By Ali Pesaranghader University of Ottawa, Ontario, Canada E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com 
--- Original Paper: Pesaranghader, Ali, and Herna L. Viktor. "Fast hoeffding drift detection method for evolving data streams." Published in: Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Springer International Publishing, 2016. URL: https://link.springer.com/chapter/10.1007/978-3-319-46227-1_7 

Copyright (C) 2022 Johannes Haug. 



---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/fhddm.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FHDDM`
FHDDM change detector. 

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/fhddm.py#L23"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `FHDDM.__init__`

```python
__init__(n: int = 100, delta: float = 1e-06, reset_after_drift: bool = False)
```

Inits the change detector. 



**Args:**
 
 - <b>`n`</b>:  Todo (left unspecified by the Tornado library). 
 - <b>`delta`</b>:  Todo (left unspecified by the Tornado library). 
 - <b>`reset_after_drift`</b>:  A boolean indicating if the change detector will be reset after a drift was detected. 




---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/fhddm.py#L66"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `FHDDM.detect_change`

```python
detect_change() → bool
```

Detects global concept drift. 



**Returns:**
 
 - <b>`bool`</b>:  True, if a concept drift was detected, False otherwise. 

---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/fhddm.py#L74"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `FHDDM.detect_partial_change`

```python
detect_partial_change() → Tuple[bool, list]
```

Detects partial concept drift. 



**Notes:**

> FHDDM does not detect partial change. 

---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/fhddm.py#L82"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `FHDDM.detect_warning_zone`

```python
detect_warning_zone() → bool
```

Detects a warning zone. 



**Notes:**

> FHDDM does not raise warnings. 

---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/fhddm.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `FHDDM.partial_fit`

```python
partial_fit(pr_scores: List[bool])
```

Updates the change detector. 



**Args:**
  pr_scores:  A boolean vector indicating correct predictions. 'True' values indicate that the prediction by the  online learner was correct, otherwise the vector contains 'False'. 

---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/fhddm.py#L40"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `FHDDM.reset`

```python
reset()
```

Resets the change detector. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
