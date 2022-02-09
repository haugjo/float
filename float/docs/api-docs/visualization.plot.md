<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/visualization/plot.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `visualization.plot`
Standard Line Plot. 

This function returns a line plot using the style and coloring of the float framework. 

Copyright (C) 2022 Johannes Haug. 


---

<a href="https://github.com/haugjo/float/tree/main/float/visualization/plot.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot`

```python
plot(
    measures: List[list],
    legend_labels: List[str],
    y_label: str,
    fig_size: tuple = (13, 5),
    font_size: int = 16,
    x_label: str = 'Time Step $t$',
    variance_measures: Optional[List[list]] = None
) → Axes
```

Returns a line plot. 

Each list provided in the measures attribute is displayed as one line. 



**Args:**
 
 - <b>`measures`</b>:  A list of lists, where each list corresponds to a series of measurements. 
 - <b>`legend_labels`</b>:  Labels for each list of measurements. These labels will be used in the legend. 
 - <b>`y_label`</b>:  The y-axis label text (e.g. the name of the performance measure that is displayed). 
 - <b>`fig_size`</b>:  The figure size (length x height) 
 - <b>`font_size`</b>:  The font size of the axis labels. 
 - <b>`x_label`</b>:  The x-axis label text. This defaults to 'Time Step t'. 
 - <b>`variance_measures`</b>:  Optionally, one can depict variances (as shaded areas around the line plot). This parameter must have the  same dimensionality as 'measures'.



**Returns:**
 
 - <b>`Axes`</b>:  The Axes object containing the line plot. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
