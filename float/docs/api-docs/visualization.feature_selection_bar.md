<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/visualization/feature_selection_bar.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `visualization.feature_selection_bar`
Feature Selection Bar Plot. 

This function returns a special bar plot that illustrates the selected features of one or multiple online feature selection methods. 

Copyright (C) 2022 Johannes Haug. 


---

<a href="https://github.com/haugjo/float/tree/main/float/visualization/feature_selection_bar.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `feature_selection_bar`

```python
feature_selection_bar(
    selected_features: List[list],
    model_names: List[str],
    feature_names: list,
    top_n_features: Optional[int] = None,
    fig_size: tuple = (13, 5),
    font_size: int = 16
) → Axes
```

Returns a bar plot that shows the number of times a feature was selected for the specified models. 



**Args:**
  selected_features:  A list of lists, where each list corresponds to the selected feature vectors of one feature selection model. 
 - <b>`model_names`</b>:  Names of the feature selection models. These labels will be used in the legend. 
 - <b>`feature_names`</b>:  The names of all input features. The feature names will be used as x-tick labels. top_n_features:  Specifies the top number of features to be displayed. If the attribute is None, we show all features in  their original order. If the attribute is not None, we select the top features of the first provided model  and compare it with the remaining models. 
 - <b>`fig_size`</b>:  The figure size (length x height) 
 - <b>`font_size`</b>:  The font size of the axis labels. 



**Returns:**
 
 - <b>`Axes`</b>:  The Axes object containing the bar plot. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
