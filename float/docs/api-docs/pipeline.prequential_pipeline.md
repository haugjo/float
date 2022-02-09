<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/pipeline/prequential_pipeline.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `pipeline.prequential_pipeline`
Prequential Pipeline Module. 

This module implements a pipeline following the prequential (i.e. test-then-train) evaluation strategy. 

Copyright (C) 2022 Johannes Haug. 



---

<a href="https://github.com/haugjo/float/tree/main/float/pipeline/prequential_pipeline.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `PrequentialPipeline`
Pipeline class for prequential evaluation. 

<a href="https://github.com/haugjo/float/tree/main/float/pipeline/prequential_pipeline.py#L24"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `PrequentialPipeline.__init__`

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
 - <b>`n_max`</b>:  Maximum number of observations used in the evaluation. label_delay_range:  The min and max delay in the availability of labels in time steps. The delay is sampled uniformly from  this range. estimate_memory_alloc:  Boolean that indicates if the method-wise change in allocated memory (GB) shall be monitored.  Note that this delivers only an indication of the approximate memory consumption and can significantly  increase the total run time of the pipeline. 
 - <b>`random_state`</b>:  A random integer seed used to specify a random number generator. 




---

<a href="https://github.com/haugjo/float/tree/main/float/pipeline/prequential_pipeline.py#L74"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `PrequentialPipeline.run`

```python
run()
```

Runs the pipeline. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
