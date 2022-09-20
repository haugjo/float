<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main\float\prediction\dynamic_model_tree.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `prediction.dynamic_model_tree`
Dynamic Model Tree Classifier. 

This module contains an implementation of the Dynamic Model Tree classification framework proposed in:

Haug, Johannes; Broelemann, Klaus; Kasneci, Gjergji. Dynamic Model Tree for Interpretable Data Stream Learning. In: 38th IEEE International Conference on Data Engineering, DOI: [10.1109/ICDE53745.2022.00237](https://doi.org/10.1109/ICDE53745.2022.00237), 2022. 

Copyright (C) 2022 Johannes Haug. 



---

<a href="https://github.com/haugjo/float/tree/main\float\prediction\dynamic_model_tree.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DynamicModelTreeClassifier`
Dynamic Model Tree Classifier. 

This implementation of the DMT uses linear (logit) simple models and the negative log likelihood loss (as described in the corresponding paper). 



**Attributes:**
 
 - <b>`classes`</b> (List):  List of the target classes. 
 - <b>`learning_rate`</b> (float):  Learning rate of the linear models. 
 - <b>`penalty_term`</b> (float):  Regularization term for the linear model (0 = no regularization penalty). 
 - <b>`penalty`</b> (str): String identifier of the type of regularization used by the linear model.  Either 'l1', 'l2', or 'elasticnet' (see documentation of sklearn SGDClassifier). 
 - <b>`epsilon`</b> (float):  Threshold required before attempting to split or prune based on the Akaike Information Criterion.  The smaller the epsilon-threshold, the stronger the evidence for splitting/pruning must be. Choose  0 < epsilon <= 1. 
 - <b>`n_saved_candidates`</b> (int):  Max. number of saved split candidates per node. 
 - <b>`p_replaceable_candidates`</b> (float):  Max. percent of saved split candidates that can be replaced by new/better candidates per training iteration. 
 - <b>`cat_features`</b> (List[int]):  List of indices (pos. in the feature vector) corresponding to categorical features. 
 - <b>`root`</b> (Node):  Root node of the Dynamic Model Tree. 

<a href="https://github.com/haugjo/float/tree/main\float\prediction\dynamic_model_tree.py#L44"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DynamicModelTreeClassifier.__init__`

```python
__init__(
    classes: List,
    learning_rate: float = 0.05,
    penalty_term: float = 0,
    penalty: str = 'l2',
    epsilon: float = 1e-07,
    n_saved_candidates: int = 100,
    p_replaceable_candidates: float = 0.5,
    cat_features: Optional[List[int]] = None,
    reset_after_drift: Optional[bool] = False
)
```

Inits the DMT. 



**Args:**
 
 - <b>`classes`</b>:  List of the target classes. 
 - <b>`learning_rate`</b>:  Learning rate of the linear models. 
 - <b>`penalty_term`</b>:  Regularization term for the linear model (0 = no regularization penalty). 
 - <b>`penalty`</b>:  String identifier of the type of regularization used by the linear model.  Either 'l1', 'l2', or 'elasticnet' (see documentation of sklearn SGDClassifier). 
 - <b>`epsilon`</b>:  Threshold required before attempting to split or prune based on the Akaike Information Criterion.  The smaller the epsilon-threshold, the stronger the evidence for splitting/pruning must be. Choose  0 < epsilon <= 1. 
 - <b>`n_saved_candidates`</b>:  Max. number of saved split candidates per node. 
 - <b>`p_replaceable_candidates`</b>:  Max. percent of saved split candidates that can be replaced by new/better candidates per training iteration. 
 - <b>`cat_features`</b>:  List of indices (pos. in the feature vector) corresponding to categorical features. 
 - <b>`reset_after_drift`</b>:  A boolean indicating if the predictor will be reset after a drift was detected. Note that the DMT  automatically adjusts to concept drift and thus generally need not be retrained. 




---

<a href="https://github.com/haugjo/float/tree/main\float\prediction\dynamic_model_tree.py#L161"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DynamicModelTreeClassifier.n_nodes`

```python
n_nodes() → Tuple[int, int, int]
```

Returns the number of nodes, leaves and the depth of the DMT. 



**Returns:**
 
 - <b>`int`</b>:  Total number of nodes. 
 - <b>`int`</b>:  Total number of leaves. 
 - <b>`int`</b>:  Depth (where a single root node has depth = 1). 

---

<a href="https://github.com/haugjo/float/tree/main\float\prediction\dynamic_model_tree.py#L92"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DynamicModelTreeClassifier.partial_fit`

```python
partial_fit(
    X: Union[numpy.__array_like._SupportsArray[numpy.dtype], numpy.__nested_sequence._NestedSequence[numpy.__array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy.__nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]],
    y: Union[numpy.__array_like._SupportsArray[numpy.dtype], numpy.__nested_sequence._NestedSequence[numpy.__array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy.__nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]]
)
```

Updates the predictor. 



**Args:**
 
 - <b>`X`</b>:  Array/matrix of observations. 
 - <b>`y`</b>:  Array of corresponding labels. 

---

<a href="https://github.com/haugjo/float/tree/main\float\prediction\dynamic_model_tree.py#L120"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DynamicModelTreeClassifier.predict`

```python
predict(
    X: Union[numpy.__array_like._SupportsArray[numpy.dtype], numpy.__nested_sequence._NestedSequence[numpy.__array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy.__nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]]
) → Union[numpy.__array_like._SupportsArray[numpy.dtype], numpy.__nested_sequence._NestedSequence[numpy.__array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy.__nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]]
```

Predicts the target values. 



**Args:**
 
 - <b>`X`</b>:  Array/matrix of observations. 



**Returns:**
 
 - <b>`ArrayLike`</b>:  Predicted labels for all observations. 

---

<a href="https://github.com/haugjo/float/tree/main\float\prediction\dynamic_model_tree.py#L138"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DynamicModelTreeClassifier.predict_proba`

```python
predict_proba(
    X: Union[numpy.__array_like._SupportsArray[numpy.dtype], numpy.__nested_sequence._NestedSequence[numpy.__array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy.__nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]]
) → Union[numpy.__array_like._SupportsArray[numpy.dtype], numpy.__nested_sequence._NestedSequence[numpy.__array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy.__nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]]
```

Predicts the probability of target values. 



**Args:**
 
 - <b>`X`</b>:  Array/matrix of observations. 



**Returns:**
 
 - <b>`ArrayLike`</b>:  Predicted probability per class label for all observations. 

---

<a href="https://github.com/haugjo/float/tree/main\float\prediction\dynamic_model_tree.py#L156"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DynamicModelTreeClassifier.reset`

```python
reset()
```

Resets the predictor. 


---

<a href="https://github.com/haugjo/float/tree/main\float\prediction\dynamic_model_tree.py#L209"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Node`
Node of the Dynamic Model Tree. 



**Attributes:**
 
 - <b>`classes`</b> (List):  List of the target classes. 
 - <b>`n_features`</b> (int):  Number of input features. 
 - <b>`learning_rate`</b> (float):  Learning rate of the linear models. 
 - <b>`penalty_term`</b> (float):  Regularization term for the linear model (0 = no regularization penalty). 
 - <b>`penalty`</b> (str): String identifier of the type of regularization used by the linear model.  Either 'l1', 'l2', or 'elasticnet' (see documentation of sklearn SGDClassifier). 
 - <b>`epsilon`</b> (float):  Threshold required before attempting to split or prune based on the Akaike Information Criterion.  The smaller the epsilon-threshold, the stronger the evidence for splitting/pruning must be. Choose  0 < epsilon <= 1. 
 - <b>`n_saved_candidates`</b> (int):  Max. number of saved split candidates per node. 
 - <b>`p_replaceable_candidates`</b> (float):  Max. percent of saved split candidates that can be replaced by new/better candidates per training iteration. 
 - <b>`cat_features`</b> (List[int]):  List of indices (pos. in the feature vector) corresponding to categorical features. 
 - <b>`linear_model`</b> (Any):  Linear (logit) model trained at the node. 
 - <b>`log_likelihood`</b> (ArrayLike):  Log-likelihood given observations that reached the node. 
 - <b>`counts_left`</b> (dict):  Number of observations per split candidate falling to the left child. 
 - <b>`log_likelihoods_left`</b> (dict):  Log-likelihoods of the left child per split candidate. 
 - <b>`gradients_left`</b> (dict):  Gradients of the left child per split candidate. 
 - <b>`counts_right`</b> (dict):  Number of observations per split candidate falling to the right child. 
 - <b>`log_likelihoods_right`</b> (dict):  Log-likelihoods of the right child per split candidate. 
 - <b>`gradients_right`</b> (dict):  Gradients of the right child per split candidate 
 - <b>`children`</b> (List[Node]):  List of child nodes. 
 - <b>`split`</b> (tuple):  Feature/value combination used for splitting. 
 - <b>`is_leaf`</b> (bool):  Indicator of whether the node is a leaf. 

<a href="https://github.com/haugjo/float/tree/main\float\prediction\dynamic_model_tree.py#L240"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Node.__init__`

```python
__init__(
    classes: List,
    n_features: int,
    learning_rate: float,
    penalty_term: float,
    penalty: str,
    epsilon: float,
    n_saved_candidates: int,
    p_replaceable_candidates: float,
    cat_features: List[int]
)
```

Inits Node. 



**Args:**
 
 - <b>`classes`</b>:  List of the target classes. 
 - <b>`n_features`</b>:  Number of input features. 
 - <b>`learning_rate`</b>:  Learning rate of the linear models. 
 - <b>`penalty_term`</b>:  Regularization term for the linear model (0 = no regularization penalty). 
 - <b>`penalty`</b>:  String identifier of the type of regularization used by the linear model.  Either 'l1', 'l2', or 'elasticnet' (see documentation of sklearn SGDClassifier). 
 - <b>`epsilon`</b>:  Threshold required before attempting to split or prune based on the Akaike Information Criterion.  The smaller the epsilon-threshold, the stronger the evidence for splitting/pruning must be. Choose  0 < epsilon <= 1. 
 - <b>`n_saved_candidates`</b>:  Max. number of saved split candidates per node. 
 - <b>`p_replaceable_candidates`</b>:  Max. percent of saved split candidates that can be replaced by new/better candidates per training iteration. 
 - <b>`cat_features`</b>:  List of indices (pos. in the feature vector) corresponding to categorical features. 




---

<a href="https://github.com/haugjo/float/tree/main\float\prediction\dynamic_model_tree.py#L671"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Node.predict_observation`

```python
predict_observation(
    x: Union[numpy.__array_like._SupportsArray[numpy.dtype], numpy.__nested_sequence._NestedSequence[numpy.__array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy.__nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]],
    get_prob: bool = False
) → Union[numpy.__array_like._SupportsArray[numpy.dtype], numpy.__nested_sequence._NestedSequence[numpy.__array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy.__nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]]
```

Predicts one observation (recurrent function). 

Passes an observation down the tree until a leaf is reached. Makes prediction at leaf. 



**Args:**
 
 - <b>`x`</b>:  Observation. 
 - <b>`get_prob`</b>:  Indicator whether to return class probabilities. 



**Returns:**
 
 - <b>`ArrayLike`</b>:  Predicted class label/probability of the given observation. 

---

<a href="https://github.com/haugjo/float/tree/main\float\prediction\dynamic_model_tree.py#L299"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Node.update`

```python
update(
    X: Union[numpy.__array_like._SupportsArray[numpy.dtype], numpy.__nested_sequence._NestedSequence[numpy.__array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy.__nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]],
    y: Union[numpy.__array_like._SupportsArray[numpy.dtype], numpy.__nested_sequence._NestedSequence[numpy.__array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy.__nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]]
)
```

Updates the node and all descendants. 

Update the parameters of the weak model at the given node. If the node is an inner node, we attempt to split on a different feature or replace the inner node by a leaf and thereby prune all previous children/subbranches. If the node is a leaf node, we attempt to split. 



**Args:**
 
 - <b>`X`</b>:  Array/matrix of observations. 
 - <b>`y`</b>:  Array of corresponding labels. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
