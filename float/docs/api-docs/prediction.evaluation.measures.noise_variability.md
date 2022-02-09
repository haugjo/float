<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/prediction/evaluation/measures/noise_variability.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `prediction.evaluation.measures.noise_variability`
Noise Variability Measure. 

This function returns the noise variability of a predictor. This measure corresponds to the mean difference of a performance measure when perturbing the input with noise. It is hence an indication of a predictor's stability under noisy inputs. 

Copyright (C) 2022 Johannes Haug. 


---

<a href="https://github.com/haugjo/float/tree/main/float/prediction/evaluation/measures/noise_variability.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `noise_variability`

```python
noise_variability(
    y_true: Union[numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]],
    y_pred: Union[numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]],
    X: Union[numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]],
    predictor: float.prediction.base_predictor.BasePredictor,
    reference_measure: Callable = <function zero_one_loss at 0x11d1669d0>,
    reference_measure_kwargs: Optional[dict] = None,
    cont_noise_loc: float = 0,
    cont_noise_scale: float = 0.1,
    cat_features: Optional[list] = None,
    cat_noise_dist: Optional[List[list]] = None,
    n_samples: int = 10,
    rng: numpy.random._generator.Generator = Generator(PCG64) at 0x128EB4040
) → float
```

Calculates the variability of a predictor under input noise. 



**Args:**
 
 - <b>`y_true`</b>:  True target labels. 
 - <b>`y_pred`</b>:  Predicted target labels. 
 - <b>`X`</b>:  Array/matrix of observations. 
 - <b>`predictor`</b>:  Predictor object. 
 - <b>`reference_measure`</b>:  Evaluation measure function. 
 - <b>`reference_measure_kwargs`</b>:  Keyword arguments of the reference measure. 
 - <b>`cont_noise_loc`</b>:  Location (mean) of a normal distribution from which we sample noise for continuous features. 
 - <b>`cont_noise_scale`</b>:  Scale (variance) of a normal distribution from which we sample noise for continuous features. 
 - <b>`cat_features`</b>:  List of indices that correspond to categorical features. 
 - <b>`cat_noise_dist`</b>:  List of lists, where each list contains the noise values of one categorical feature. 
 - <b>`n_samples`</b>:  Number of times we sample noise and investigate divergence from the original loss. 
 - <b>`rng`</b>:  A numpy random number generator object. The global random state of the pipeline will be used to this end. 



**Returns:**
 
 - <b>`float`</b>:  Mean difference to the original loss for n input perturbations. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
