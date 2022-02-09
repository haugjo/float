<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/mddm_g.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `change_detection.tornado.mddm_g`
McDiarmid Drift Detection Method (Geometric Scheme). 

The source code was adopted from https://github.com/alipsgh/tornado, please cite: The Tornado Framework By Ali Pesaranghader University of Ottawa, Ontario, Canada E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com 
--- Original Paper: Pesaranghader, Ali, et al. "McDiarmid Drift Detection Method for Evolving Data Streams." Published in: International Joint Conference on Neural Network (IJCNN 2018) URL: https://arxiv.org/abs/1710.02030 

Copyright (C) 2022 Johannes Haug. 



---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/mddm_g.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MDDMG`
MDDMG change detector. 

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/mddm_g.py#L23"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `MDDMG.__init__`

```python
__init__(
    n: int = 100,
    ratio: float = 1.01,
    delta: float = 1e-06,
    reset_after_drift: bool = False
)
```

Inits the change detector. 



**Args:**
 
 - <b>`n`</b>:  Todo (left unspecified by the Tornado library). 
 - <b>`ratio`</b>:  Todo (left unspecified by the Tornado library). 
 - <b>`delta`</b>:  Todo (left unspecified by the Tornado library). 
 - <b>`reset_after_drift`</b>:  A boolean indicating if the change detector will be reset after a drift was detected. 




---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/mddm_g.py#L67"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `MDDMG.detect_change`

```python
detect_change() → bool
```

Detects global concept drift. 



**Returns:**
 
 - <b>`bool`</b>:  True, if a concept drift was detected, False otherwise. 

---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/mddm_g.py#L75"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `MDDMG.detect_partial_change`

```python
detect_partial_change() → Tuple[bool, list]
```

Detects partial concept drift. 



**Notes:**

> MDDMG does not detect partial change. 

---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/mddm_g.py#L83"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `MDDMG.detect_warning_zone`

```python
detect_warning_zone() → bool
```

Detects a warning zone. 



**Notes:**

> MDDMG does not raise warnings. 

---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/mddm_g.py#L47"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `MDDMG.partial_fit`

```python
partial_fit(pr_scores: List[bool])
```

Updates the change detector. 



**Args:**
  pr_scores:  A boolean vector indicating correct predictions. 'True' values indicate that the prediction by the  online learner was correct, otherwise the vector contains 'False'. 

---

<a href="https://github.com/haugjo/float/tree/main/float/change_detection/tornado/mddm_g.py#L42"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `MDDMG.reset`

```python
reset()
```

Resets the change detector. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
