<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/visualization/change_detection_scatter.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `visualization.change_detection_scatter`
Change Detection Scatter Plot. 

This function returns a special scatter plot that illustrates the detected concept drifts of one or multiple change detection methods. 

Copyright (C) 2022 Johannes Haug. 


---

<a href="https://github.com/haugjo/float/tree/main/float/visualization/change_detection_scatter.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `change_detection_scatter`

```python
change_detection_scatter(
    detected_drifts: List[list],
    model_names: List[str],
    n_samples: int,
    known_drifts: Union[List[int], List[tuple]],
    batch_size: int,
    n_pretrain: int,
    fig_size: tuple = (13, 5),
    font_size: int = 16
) → Axes
```

Returns a scatter plot showing the known and the detected concept drifts of the provided models. 



**Args:**
  detected_drifts:  A list of lists, where each list corresponds the detected drifts of one concept drift detector. 
 - <b>`model_names`</b>:  Names of the concept drift detection models. These labels will be used in the legend. 
 - <b>`n_samples`</b>:  The total number of samples observed. known_drifts (List[int] | List[tuple]):  The positions in the dataset (indices) corresponding to known concept drifts. batch_size:  The batch size used for the evaluation of the data stream. This is needed to translate the known drift  positions to logical time steps (which is the format of the detected drifts). n_pretrain:  The number of observations used for pre-training. This number needs to be subtracted from the known drift  positions in order to translate them to the correct logical time steps. 
 - <b>`fig_size`</b>:  The figure size (length x height) 
 - <b>`font_size`</b>:  The font size of the axis labels. 



**Returns:**
 
 - <b>`Axes`</b>:  The Axes object containing the plot. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
