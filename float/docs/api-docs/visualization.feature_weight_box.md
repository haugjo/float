<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/visualization/feature_weight_box.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `visualization.feature_weight_box`
Feature Weight Box Plot. 

This function returns a box plot that illustrates the feature weights of one or multiple online feature selection methods. 

Copyright (C) 2022 Johannes Haug. 


---

<a href="https://github.com/haugjo/float/tree/main/float/visualization/feature_weight_box.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `feature_weight_box`

```python
feature_weight_box(
    feature_weights: List[list],
    model_names: List[str],
    feature_names: list,
    top_n_features: Optional[int] = None,
    fig_size: tuple = (13, 5),
    font_size: int = 16
) → Axes
```

Returns a box plot that shows the distribution of weights for the selected or all features. 



**Args:**

 - <b>`feature_weights`</b>:  A list of lists, where each list corresponds to the feature weights of one feature selection model. 
 - <b>`model_names`</b>:  Names of the feature selection models. These labels will be used in the legend. 
 - <b>`feature_names`</b>:  The names of all input features. The feature names will be used as x-tick labels. 
 - <b>`top_n_features`</b>:  Specifies the top number of features to be displayed. If the attribute is None, we show all features in  their original order. If the attribute is not None, we select the top features according to their median  value. 
 - <b>`fig_size`</b>:  The figure size (length x height) 
 - <b>`font_size`</b>:  The font size of the axis labels. 



**Returns:**
 
 - <b>`Axes`</b>:  The Axes object containing the bar plot. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
