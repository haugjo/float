<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/cusum.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `change_detection.tornado.cusum`
Cumulative Sum Drift Detection Method. 

The source code was adopted from https://github.com/alipsgh/tornado, please cite: The Tornado Framework By Ali Pesaranghader University of Ottawa, Ontario, Canada E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com 
--- Original Paper: Page, Ewan S. "Continuous inspection schemes." Published in: Biometrika 41.1/2 (1954): 100-115. URL: http://www.jstor.org/stable/2333009 

Copyright (C) 2022 Johannes Haug. 



---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/cusum.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Cusum`
Cusum Change Detector. 

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/cusum.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Cusum.__init__`

```python
__init__(
    min_instance: int = 30,
    delta: float = 0.005,
    lambda_: int = 50,
    reset_after_drift: bool = False
)
```

Inits the change detector. 



**Args:**
 
 - <b>`min_instance`</b>:  Todo (left unspecified by the Tornado library). 
 - <b>`delta`</b>:  Todo (left unspecified by the Tornado library). 
 - <b>`lambda_`</b>:  Todo (left unspecified by the Tornado library). 
 - <b>`reset_after_drift`</b>:  A boolean indicating if the change detector will be reset after a drift was detected. 




---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/cusum.py#L71"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Cusum.detect_change`

```python
detect_change() → bool
```

Detects global concept drift. 



**Returns:**
 
 - <b>`bool`</b>:  True, if a concept drift was detected, False otherwise. 

---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/cusum.py#L79"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Cusum.detect_partial_change`

```python
detect_partial_change() → Tuple[bool, list]
```

Detects partial concept drift. 



**Notes:**

> Cusum does not detect partial change. 

---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/cusum.py#L87"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Cusum.detect_warning_zone`

```python
detect_warning_zone() → bool
```

Detects a warning zone. 



**Notes:**

> Cusum does not raise warnings. 

---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/cusum.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Cusum.partial_fit`

```python
partial_fit(pr_scores: List[bool])
```

Updates the change detector. 



**Args:**
  pr_scores:  A boolean vector indicating correct predictions. 'True' values indicate that the prediction by the  online learner was correct, otherwise the vector contains 'False'. 

---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/cusum.py#L42"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Cusum.reset`

```python
reset()
```

Resets the change detector. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
