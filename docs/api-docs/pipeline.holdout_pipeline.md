<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/pipeline/holdout_pipeline.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `pipeline.holdout_pipeline`
Periodic Holdout Pipeline. 

This module implements a pipeline following the periodic holdout evaluation strategy. 

Copyright (C) 2022 Johannes Haug. 



---

<a href="https://github.com/haugjo/float/tree/main/float/pipeline/holdout_pipeline.py#L23"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `HoldoutPipeline`
Pipeline class for periodic holdout evaluation. 



**Attributes:**

- <b>`test_set`</b> (Tuple[ArrayLike, ArrayLike] | None):  A tuple containing the initial test observations and labels used for the holdout evaluation.  
- <b>`test_replace_interval`</b> (int | None):  This integer specifies in which interval we replace the oldest test observation. For example, if  test_replace_interval=10 then we will use every 10th observation to replace the currently oldest test  observation. Note that test observations will not be used for training, hence this interval should not  be chosen too small. If argument is None, we use the complete batch at testing time in the evluation. 

<a href="https://github.com/haugjo/float/tree/main/float/pipeline/holdout_pipeline.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `HoldoutPipeline.__init__`

```python
__init__(
    data_loader: float.data.data_loader.DataLoader,
    predictor: Union[float.prediction.base_predictor.BasePredictor, List[float.prediction.base_predictor.BasePredictor]],
    prediction_evaluator: float.prediction.evaluation.prediction_evaluator.PredictionEvaluator,
    change_detector: Optional[float.change_detection.base_change_detector.BaseChangeDetector] = None,
    change_detection_evaluator: Optional[float.change_detection.evaluation.change_detection_evaluator.ChangeDetectionEvaluator] = None,
    feature_selector: Optional[float.feature_selection.base_feature_selector.BaseFeatureSelector] = None,
    feature_selection_evaluator: Optional[float.feature_selection.evaluation.feature_selection_evaluator.FeatureSelectionEvaluator] = None,
    batch_size: int = 1,
    n_pretrain: int = 100,
    n_max: int = inf,
    label_delay_range: Optional[tuple] = None,
    test_set: Optional[Tuple[Union[numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]], Union[numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]]]] = None,
    test_interval: int = 10,
    test_replace_interval: Optional[int] = None,
    estimate_memory_alloc: bool = False,
    random_state: int = 0
)
```

Initializes the pipeline. 

**Args:**
 
 - <b>`data_loader`</b>:  Data loader object. 
 - <b>`predictor`</b>:  Predictor object or list of predictor objects. 
 - <b>`prediction_evaluator`</b>:  Evaluator object for the predictive model(s). 
 - <b>`change_detector`</b>:  Concept drift detection model. 
 - <b>`change_detection_evaluator`</b>:  Evaluator for active concept drift detection. 
 - <b>`feature_selector`</b>:  Online feature selection model. 
 - <b>`feature_selection_evaluator`</b>:  Evaluator for the online feature selection. 
 - <b>`batch_size`</b>:  Batch size, i.e. no. of observations drawn from the data loader at one time step. 
 - <b>`n_pretrain`</b>:  Number of observations used for the initial training of the predictive model. 
 - <b>`n_max`</b>:  Maximum number of observations used in the evaluation. 
 - <b>`label_delay_range`</b>:  The min and max delay in the availability of labels in time steps. The delay is sampled uniformly from  this range. 
 - <b>`test_set`</b>:  A tuple containing the initial test observations and labels used for the holdout evaluation. 
 - <b>`test_interval`</b>:  The interval/frequency at which the online learning models are evaluated. 
 - <b>`test_replace_interval`</b>:  This integer specifies in which interval we replace the oldest test observation. For example, if  test_replace_interval=10 then we will use every 10th observation to replace the currently oldest test  observation. Note that test observations will not be used for training, hence this interval should not  be chosen too small. If argument is None, we use the complete batch at testing time in the evaluation. 
 - <b>`estimate_memory_alloc`</b>:  Boolean that indicates if the method-wise change in allocated memory (GB) shall be monitored.  Note that this delivers only an indication of the approximate memory consumption and can significantly  increase the total run time of the pipeline. 
 - <b>`random_state`</b>:  A random integer seed used to specify a random number generator. 




---

<a href="https://github.com/haugjo/float/tree/main/float/pipeline/holdout_pipeline.py#L108"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `HoldoutPipeline.run`

```python
run()
```

Runs the pipeline. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
