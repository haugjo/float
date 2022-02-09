<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/pipeline/distributed_fold_pipeline.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `pipeline.distributed_fold_pipeline`
Distributed Fold Pipeline. 

This module contains a pipeline that performs a k-fold distributed validation as proposed by Albert Bifet, Gianmarco de Francisci Morales, Jesse Read, Geoff Holmes, and Bernhard Pfahringer. 2015. Efficient Online Evaluation of Big Data Stream Classifiers. In Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '15). Association for Computing Machinery, New York, NY, USA, 59–68. 

The distributed fold pipeline maintains multiple parallel instances of each provided predictor object and enables more robust and statistically significant results. The following three modes defined by Bifet et al. are implemented: 

1. k-fold distributed cross-validation: each example is used for testing in one classifier instance selected randomly, and used for training by all the others. 

2. k-fold distributed split-validation: each example is used for training in one classifier instance selected randomly, and for testing in the other classifiers. 

3. k-fold distributed bootstrap validation: each example is used for training in each classifier instance according to a weight from a Poisson(1) distribution. This results in each example being used for training in approximately two thirds of the classifiers instances,  with a separate weight in each classifier, and for testing in the rest. 

Copyright (C) 2022 Johannes Haug. 



---

<a href="https://github.com/haugjo/float/tree/main/float/pipeline/distributed_fold_pipeline.py#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DistributedFoldPipeline`
Pipeline class for a k-fold distributed evaluation. 



**Attributes:**

 - <b>`validation_mode`</b> (str):  A string indicating the k-fold distributed validation mode to use. One of 'cross', 'batch' and 'bootstrap'. 
 - <b>`n_parallel_instances`</b> (int):  The number of instances of the specified predictor that will be trained in parallel. 
 - <b>`n_unique_predictors`</b> (int):  The number of predictor objects originally specified. 
 - <b>`predictors`</b> (List[BasePredictor]):  All instances of the predictive model(s). 
 - <b>`prediction_evaluators`</b> (List[PredictionEvaluator]):  Evaluator(s) for all instances of the predictive model(s). 

<a href="https://github.com/haugjo/float/tree/main/float/pipeline/distributed_fold_pipeline.py#L50"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DistributedFoldPipeline.__init__`

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
    validation_mode: str = 'cross',
    n_parallel_instances: int = 2,
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
 - <b>`validation_mode`</b>:  A string indicating the k-fold distributed validation mode to use. One of 'cross', 'split' and  'bootstrap'. 
 - <b>`n_parallel_instances`</b>:  The number of instances of the specified predictor that will be trained in parallel. 
 - <b>`estimate_memory_alloc`</b>:  Boolean that indicates if the method-wise change in allocated memory (GB) shall be monitored.  Note that this delivers only an indication of the approximate memory consumption and can significantly  increase the total run time of the pipeline. 
 - <b>`random_state`</b>:  A random integer seed used to specify a random number generator. 




---

<a href="https://github.com/haugjo/float/tree/main/float/pipeline/distributed_fold_pipeline.py#L121"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DistributedFoldPipeline.run`

```python
run()
```

Runs the pipeline. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
