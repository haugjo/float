<!-- markdownlint-disable -->

<a href="https://github.com/haugjo/float/tree/main/float/pipeline/utils_pipeline.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `pipeline.utils_pipeline`
Pipeline Utils. 

This module contains utility functions for the Pipeline Module. In particular, this module contains functionality to validate provided attributes, update the console progress bar and print a final summary of the evaluation run. 

Copyright (C) 2022 Johannes Haug. 

---

<a href="https://github.com/haugjo/float/tree/main/float/pipeline/utils_pipeline.py#L34"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `validate_pipeline_attrs`

```python
validate_pipeline_attrs(pipeline: 'BasePipeline')
```

Validates the input parameters and attributes of a pipeline obect. 



**Args:**
 
 - <b>`pipeline`</b>:  Pipeline object. 



**Raises:**
 
 - <b>`AttributeError`</b>:  If a crucial parameter to run the pipeline is missing or is invalid. 


---

<a href="https://github.com/haugjo/float/tree/main/float/pipeline/utils_pipeline.py#L143"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `update_progress_bar`

```python
update_progress_bar(pipeline: 'BasePipeline')
```

Updates the progress bar in the console after one training iteration. 



**Args:**
 
 - <b>`pipeline`</b>:  Pipeline object. 


---

<a href="https://github.com/haugjo/float/tree/main/float/pipeline/utils_pipeline.py#L165"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `print_evaluation_summary`

```python
print_evaluation_summary(pipeline: 'BasePipeline')
```

Prints a summary of the given pipeline evaluation to the console. 



**Args:**
 
 - <b>`pipeline`</b>:  Pipeline object. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
