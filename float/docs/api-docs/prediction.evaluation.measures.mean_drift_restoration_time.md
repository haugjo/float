<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/prediction/evaluation/measures/mean_drift_restoration_time.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `prediction.evaluation.measures.mean_drift_restoration_time`
Drift Restoration Time Measure. 

This function returns the mean drift restoration time, i.e. the average number of iterations (time steps) after a known concept drift, before the previous performance has been restored. It is hence a measure to quantify the adaptability of a predictor under concept drift. 

Copyright (C) 2022 Johannes Haug. 


---

<a href="https://github.com/haugjo/float/tree/main/float/prediction/evaluation/measures/mean_drift_restoration_time.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `mean_drift_restoration_time`

```python
mean_drift_restoration_time(
    result: dict,
    known_drifts: Union[List[int], List[tuple]],
    batch_size: int,
    reference_measure: Callable = <function zero_one_loss at 0x11d1669d0>,
    reference_measure_kwargs: Optional[dict] = None,
    incr: bool = False,
    interval: int = 10
) → float
```

Calculates the mean restoration time after known concept drifts. 



**Args:**
 
 - <b>`result`</b>:  A result dictionary from the PredictionEvaluator object. 
 - <b>`known_drifts`</b>:  The positions in the dataset (indices) corresponding to known concept drifts. 
 - <b>`batch_size`</b>:  The number of observations processed per iteration/time step. 
 - <b>`reference_measure`</b>:  Evaluation measure function. 
 - <b>`reference_measure_kwargs`</b>:  Keyword arguments of the reference measure. This attribute is maintained for consistency reasons, but is  not used by this performance measure. 
 - <b>`incr`</b>:  Boolean indicating whether the evaluation measure is incremental (i.e. higher is better). 
 - <b>`interval`</b>:  Scalar specifying the size of the interval (i.e. number of time steps) after known concept drift, in which  we investigate a performance decay of the reference measure. 



**Returns:**
 
 - <b>`float`</b>:  Current mean no. of iterations before recovery from (known) concept drifts. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
