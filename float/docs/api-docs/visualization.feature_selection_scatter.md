<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/visualization/feature_selection_scatter.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `visualization.feature_selection_scatter`
Feature Selection Scatter Plot. 

This function returns a special scatter plot that illustrates the selected features over time of one online feature selection method. 

Copyright (C) 2022 Johannes Haug. 


---

<a href="https://github.com/haugjo/float/tree/main/float/visualization/feature_selection_scatter.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `feature_selection_scatter`

```python
feature_selection_scatter(
    selected_features: List[list],
    fig_size: tuple = (13, 5),
    font_size: int = 16
) → Axes
```

Returns a scatter plot that illustrates the selected features over time for the specified models. 



**Args:**

 - <b>`selected_features`</b>:  A list of lists, where each list corresponds to the selected feature vector at one time step. 
 - <b>`fig_size`</b>:  The figure size (length x height) 
 - <b>`font_size`</b>:  The font size of the axis labels. 



**Returns:**
 
 - <b>`Axes`</b>:  The Axes object containing the plot. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
