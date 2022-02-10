<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/pipeline/base_pipeline.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `pipeline.base_pipeline`
Base Pipeline. 

This module contains functionality to construct a pipeline and run experiments in a standardized and modular fashion. This abstract BasePipeline class should be used as a super class for all specific evaluation pipelines. 

Copyright (C) 2022 Johannes Haug. 



---

<a href="https://github.com/haugjo/float/tree/main/float/pipeline/base_pipeline.py#L30"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BasePipeline`
Abstract base class for evaluation pipelines. 



**Attributes:**
 
 - <b>`data_loader`</b> (DataLoader):  Data loader object. 
 - <b>`predictors`</b> (List[BasePredictor]):  Predictive model(s). 
 - <b>`prediction_evaluators`</b> (List[PredictionEvaluator]):  Evaluator(s) for the predictive model(s). 
 - <b>`change_detector`</b> (ConceptDriftDetector | None):  Concept drift detection model. 
 - <b>`change_detection_evaluator`</b> (ChangeDetectionEvaluator | None):  Evaluator for active concept drift detection. 
 - <b>`feature_selector`</b> (BaseFeatureSelector | None):  Online feature selection model. 
 - <b>`feature_selection_evaluator`</b> (FeatureSelectionEvaluator | None):  Evaluator for the online feature selection. 
 - <b>`batch_size`</b> (int | None):  Batch size, i.e. no. of observations drawn from the data loader at one time step. 
 - <b>`n_pretrain`</b> (int | None):  Number of observations used for the initial training of the predictive model. 
 - <b>`n_max`</b> (int | None):  Maximum number of observations used in the evaluation. 
 - <b>`label_delay_range`</b> (tuple | None):  The min and max delay in the availability of labels in time steps. The delay is sampled uniformly from  this range. 
 - <b>`estimate_memory_alloc`</b> (bool):  Boolean that indicates if the method-wise change in allocated memory (GB) shall be monitored.  Note that this delivers only an indication of the approximate memory consumption and can significantly  increase the total run time of the pipeline. 
 - <b>`test_interval`</b> (int):  The interval/frequency at which the online learning models are evaluated. This parameter is always 1 for a  prequential or distributed fold evaluation. 
 - <b>`rng`</b> (Generator):  A numpy random number generator object. 
 - <b>`start_time`</b> (float):  Physical start time. 
 - <b>`time_step`</b> (int):  Current logical time step, i.e. iteration. 
 - <b>`n_total`</b> (int):  Total number of observations currently observed. 

<a href="https://github.com/haugjo/float/tree/main/float/pipeline/base_pipeline.py#L60"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `BasePipeline.__init__`

```python
__init__(
    data_loader: float.data.data_loader.DataLoader,
    predictor: Union[float.prediction.base_predictor.BasePredictor, List[float.prediction.base_predictor.BasePredictor]],
    prediction_evaluator: float.prediction.evaluation.prediction_evaluator.PredictionEvaluator,
    change_detector: Optional[float.change_detection.base_change_detector.BaseChangeDetector],
    change_detection_evaluator: Optional[float.change_detection.evaluation.change_detection_evaluator.ChangeDetectionEvaluator],
    feature_selector: Optional[float.feature_selection.base_feature_selector.BaseFeatureSelector],
    feature_selection_evaluator: Optional[float.feature_selection.evaluation.feature_selection_evaluator.FeatureSelectionEvaluator],
    batch_size: int,
    n_pretrain: int,
    n_max: int,
    label_delay_range: Optional[tuple],
    test_interval: int,
    estimate_memory_alloc: bool,
    random_state: int
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
 - <b>`test_interval`</b>:  The interval/frequency at which the online learning models are evaluated. This parameter is always 1 for  a prequential evaluation. 
 - <b>`estimate_memory_alloc`</b>:  Boolean that indicates if the method-wise change in allocated memory (GB) shall be monitored.  Note that this delivers only an indication of the approximate memory consumption and can significantly  increase the total run time of the pipeline. 
 - <b>`random_state`</b>:  A random integer seed used to specify a random number generator. 



**Raises:**
 
 - <b>`AttributeError`</b>:  If one of the provided objects is not valid. 




---

<a href="https://github.com/haugjo/float/tree/main/float/pipeline/base_pipeline.py#L132"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `BasePipeline.run`

```python
run()
```

Runs the pipeline. 

This function is specifically implemented for each evaluation strategy. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
