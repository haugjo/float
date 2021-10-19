<p align="center">
  <img alt="float" src="logo.png" width="400"/>
</p>
<p align="center">
    <em>[Todo: insert badges for pypi and documentation]</em>
</p>

<p align="center">
    <strong><em>float</em> (<u>f</u>riction<u>l</u>ess <u>o</u>nline model <u>a</u>nalysis and <u>t</u>esting)</strong> is a modular framework for standardized and high-quality evaluations of online learning methods.
    <br><em>float</em> can combine custom code with functionality of popular libraries and provides easy access to meaningful evaluation strategies, measures and visualisations.
</p>

>**Release Notes:**<br>
>*v0.0.1* - initial release. *float* currently supports [scikit-multiflow](https://github.com/scikit-multiflow/scikit-multiflow). We plan to offer support for [river](https://github.com/online-ml/river/) soon.

## Installation
Install float with pip (NOTE for reviewers: the package is not available on pip yet):
```
pip install float
```
Alternatively, you may also clone the repository from Github:
```
git clone git@github.com:haugjo/float.git
```

## Requirements
Float is able to run with the following package versions (note that older versions have not been tested, but may also work): Todo

## Quickstart
Below, we show how to run a basic prequential evaluation in float:
```python
from skmultiflow.trees import HoeffdingTreeClassifier
from sklearn.metrics import zero_one_loss

from float.data import DataLoader
from float.prediction.evaluation import PredictionEvaluator
from float.prediction.evaluation.measures import noise_variability
from float.pipeline import PrequentialPipeline
from float.prediction.skmultiflow import SkmultiflowClassifier

# Load a data set from main memory with the DataLoader module.
data_loader = DataLoader(file_path=f'data/datasets/spambase.csv', target_col=0)

# Set up an online classifier. Note, we need a wrapper to use scikit-multiflow functionality.
classifier = SkmultiflowClassifier(model=HoeffdingTreeClassifier(), 
                                   classes=data_loader.stream.target_values)

# Set up an evaluation object for the classifier:
# Here, we want to measure the zero_one_loss and the noise_variability as an indication of robustness.
# The arguments of the measure functions can be directly added to the Evaluator object constructor, 
# e.g. we may specify the number of samples (n_samples) used for the noise_variability measure.
evaluator = PredictionEvaluator(measure_funcs=[zero_one_loss, noise_variability], n_samples=15)

# Set up a pipeline for a prequential evaluation of the classifier.
pipeline = PrequentialPipeline(data_loader=data_loader, 
                               predictor=classifier, 
                               prediction_evaluator=evaluator, 
                               batch_size=25)

# Run the experiment.
pipeline.run()
```
```console
Output:
Pretrain the predictor with 100 observation(s).
[====================] 100%
################################## SUMMARY ##################################
Evaluation has finished after 18.879928588867188s
Data Set data/datasets/spambase.csv
The pipeline has processed 4601 instances in total, using batches of size 25.
----------------------
Prediction:
| Model                   |   Avg. Test Comp. Time |   Avg. Train Comp. Time |   Avg. zero_one_loss |   Avg. noise_variability |
|-------------------------|------------------------|-------------------------|----------------------|--------------------------|
| HoeffdingTreeClassifier |             0.00829525 |              0.00938375 |             0.179778 |                 0.232874 |
#############################################################################
```
More exemplary experiments can be found in the ./experiments folder and in the documentation that is soon to come.

## References
Please refer to the following paper when using float: _Todo_

Please also make sure that you reference the corresponding papers, when using one of the provided implementations or data sets.

## Contribution
We welcome meaningful contributions by the community. In particular, we encourage contribution of new evaluation strategies, measures and data sets.

Additionally, we welcome implementations of novel online learning models (however, we discourage contributing models that are already included in one 
of the major libraries. Again, note that float is not primarily intended as a library of state-of-the-art online learning methods, but as a toolkit for their standardised evaluation.)

All contributed source code must adhere to the following criteria:
- Code must conform with the [PEP8](https://www.python.org/dev/peps/pep-0008/) standard.
- Docstrings must conform with the [Google docstring](https://google.github.io/styleguide/pyguide.html) convention.
- All existing unit tests must run without an error (new unit tests should be provided as needed).

Please feel free to contact us, if you plan to make a contribution.
