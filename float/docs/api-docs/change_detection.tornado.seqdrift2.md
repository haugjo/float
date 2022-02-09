<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/seqdrift2.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `change_detection.tornado.seqdrift2`
SeqDrift2 Drift Detection Method. 

The source code was adopted from https://github.com/alipsgh/tornado, please cite: The Tornado Framework By Ali Pesaranghader University of Ottawa, Ontario, Canada E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com 
--- Original Paper: Pears, Russel, Sripirakas Sakthithasan, and Yun Sing Koh. "Detecting concept change in dynamic data streams." Published in: Machine Learning 97.3 (2014): 259-293. URL: https://link.springer.com/article/10.1007/s10994-013-5433-9 

Copyright (C) 2022 Johannes Haug. 



---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/seqdrift2.py#L23"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SeqDrift2`
SeqDrift2 change detector. 

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/seqdrift2.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `SeqDrift2.__init__`

```python
__init__(
    delta: float = 0.01,
    block_size: int = 200,
    reset_after_drift: bool = False
)
```

Inits the change detector. 



**Args:**
 
 - <b>`delta`</b>:  Todo (left unspecified by the Tornado library). 
 - <b>`block_size`</b>:  Todo (left unspecified by the Tornado library). 
 - <b>`reset_after_drift`</b>:  A boolean indicating if the change detector will be reset after a drift was detected. 




---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/seqdrift2.py#L55"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `SeqDrift2.detect_change`

```python
detect_change() → bool
```

Detects global concept drift. 



**Returns:**
 
 - <b>`bool`</b>:  True, if a concept drift was detected, False otherwise. 

---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/seqdrift2.py#L63"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `SeqDrift2.detect_partial_change`

```python
detect_partial_change() → Tuple[bool, list]
```

Detects partial concept drift. 



**Notes:**

> SeqDrift2 does not detect partial change. 

---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/seqdrift2.py#L71"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `SeqDrift2.detect_warning_zone`

```python
detect_warning_zone() → bool
```

Detects a warning zone. 



**Notes:**

> SeqDrift2 does not raise warnings. 

---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/seqdrift2.py#L44"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `SeqDrift2.partial_fit`

```python
partial_fit(pr_scores: List[bool])
```

Updates the change detector. 



**Args:**
  pr_scores:  A boolean vector indicating correct predictions. 'True' values indicate that the prediction by the  online learner was correct, otherwise the vector contains 'False'. 

---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/seqdrift2.py#L40"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `SeqDrift2.reset`

```python
reset()
```

Resets the change detector. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
