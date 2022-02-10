<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/feature_selection/evaluation/measures/nogueira_stability.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `feature_selection.evaluation.measures.nogueira_stability`
Nogueira Feature Set Stability Measure. 

This function returns the feature set stability measure introduced by: NOGUEIRA, Sarah; SECHIDIS, Konstantinos; BROWN, Gavin. On the stability of feature selection algorithms. J. Mach. Learn. Res., 2017, 18. Jg., Nr. 1, S. 6345-6398. 

Copyright (C) 2022 Johannes Haug. 


---

<a href="https://github.com/haugjo/float/tree/main/float/feature_selection/evaluation/measures/nogueira_stability.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `nogueira_stability`

```python
nogueira_stability(
    selected_features_history: List[list],
    n_total_features: int
) → float
```

Calculates the Nogueira measure for feature selection stability. 



**Args:**
 
 - <b>`selected_features_history`</b>:  A list of all selected feature vectors obtained over time. 
 - <b>`n_total_features`</b>:  The total number of features. 



**Returns:**
 
 - <b>`float`</b>:  The feature set stability due to Nogueira et al. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
