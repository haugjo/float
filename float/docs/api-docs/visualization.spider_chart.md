<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/visualization/spider_chart.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `visualization.spider_chart`
Spider Plot. 

This function returns a spider plot that can be used to compare models with respect to different performance measures. 

Copyright (C) 2022 Johannes Haug. 


---

<a href="https://github.com/haugjo/float/tree/main/float/visualization/spider_chart.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `spider_chart`

```python
spider_chart(
    measures: List[List],
    measure_names: List[str],
    legend_names: List[str],
    ranges: Optional[List[Tuple]] = None,
    invert: Optional[List[bool]] = None,
    fig_size: tuple = (8, 5),
    font_size: int = 16
) → Axes
```

Returns a spider chart that displays the specified measures. 



**Args:**

 - <b>`measures`</b>:  A list of lists. Each list corresponds to different (e.g. model-wise) results for one performance measure.  
 - <b>`measure_names`</b>:  The names of the performance measures that are shown. This attribute has the same length as 'measures'  
 - <b>`legend_names`</b>:  Legend labels for each different result per measure (e.g. model names). This attribute has the same length  as each list in the 'measures' attribute.  
 - <b>`ranges`</b>:  The value ranges for each of the measures. If None, the range will be set to (0,1) per default. Otherwise,  each tuple in the list corresponds to the range of the measure at the respective position in 'measures'.  
 - <b>`invert`</b>:  A list of bool values indicating for each measure if it should be inverted. We invert a measure if a  lower value is better than higher value. Otherwise, the spider chart may be confusing. If None, 'invert'  will be set to False for each measure. 
 - <b>`fig_size`</b>:  The figure size (length x height). 
 - <b>`font_size`</b>:  The font size of the axis labels. 



**Returns:**
 
 - <b>`Axes`</b>:  The Axes object containing the plot. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
