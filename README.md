<p align="center">
  <img alt="float" src="logo.png" width="400"/>
</p>
<p align="center">
    <em>[Placeholder for badges]</em>
</p>

<p align="center">
    <strong><em>float</em> (<u>f</u>riction<u>l</u>ess <u>o</u>nline model <u>a</u>nalysis and <u>t</u>esting)</strong> is a modular framework for standardized and high-quality evaluations of online learning methods.
    <br><em>float</em> can combine custom code with functionality of popular libraries and provides easy access to meaningful evaluation strategies, measures and visualisations.
</p>

>**Release Notes:**<br>
>*1st Feb. 2022* - Initial release on Github.

## Installation
<!--
Install float with pip:
```
pip install float
```
-->
Execute the following code in a command line to clone the float repository from Github:
```
git clone git@github.com:haugjo/float.git
```
*[!] A release of float via the Python Packaging Index (pypi) will follow soon.*

## Requirements
Float is supported by python versions >=3.8.x and has the following package requirements:
- numpy~=1.22.1
- scikit-learn~=1.0.2
- tabulate~=0.8.5
- matplotlib~=3.4.2
- scipy~=1.7.3
- pytorch~=1.4.0
- pandas~=1.4.0
- river~=0.9.0
- scikit-multiflow~=0.5.3

*[!] Older versions have not been tested, but might also be compatible.*

## Quickstart
Below, we show how to run a basic prequential evaluation in float. More extensive examples, 
including a use of the visualizer module, can be found in the ```./experiments``` folder.
```python
from skmultiflow.trees import HoeffdingTreeClassifier
from sklearn.metrics import zero_one_loss

from float.data import DataLoader
from float.prediction.evaluation import PredictionEvaluator
from float.prediction.evaluation.measures import noise_variability
from float.pipeline import PrequentialPipeline
from float.prediction.skmultiflow import SkmultiflowClassifier

# Load a data set from main memory with the DataLoader module.
data_loader = DataLoader(path='./float/data/datasets/spambase.csv', target_col=-1)

# Set up an online classifier. Note, we need a wrapper to use scikit-multiflow functionality.
classifier = SkmultiflowClassifier(model=HoeffdingTreeClassifier(),
                                   classes=data_loader.stream.target_values)

# Set up an evaluation object for the classifier:
# Here, we want to measure the zero_one_loss and the noise_variability as an indication of robustness.
# The arguments of the measure functions can be directly added to the Evaluator object constructor,
# e.g. we may specify the number of samples (n_samples) and the reference_measure used to compute the noise_variability.
evaluator = PredictionEvaluator(measure_funcs=[zero_one_loss, noise_variability],
                                n_samples=15,
                                reference_measure=zero_one_loss)

# Set up a pipeline for a prequential evaluation of the classifier.
pipeline = PrequentialPipeline(data_loader=data_loader,
                               predictor=classifier,
                               prediction_evaluator=evaluator,
                               n_max=data_loader.stream.n_samples,
                               batch_size=25)

# Run the experiment.
pipeline.run()
```
```console
Output:
[====================] 100%
################################## SUMMARY ##################################
Evaluation has finished after 9.49581503868103s
Data Set: ./float/data/datasets/spambase.csv
The PrequentialPipeline has processed 4601 instances, using batches of size 25.
-------------------------------------------------------------------------
*** Prediction ***
Model: SkmultiflowClassifier.HoeffdingTreeClassifier
| Performance Measure    |     Value |
|------------------------|-----------|
| Avg. Test Comp. Time   | 0.0032578 |
| Avg. Train Comp. Time  | 0.0053197 |
| Avg. zero_one_loss     | 0.180552  |
| Avg. noise_variability | 0.241901  |
#############################################################################
```
## Datasets
float is accompanied by a small set of popular streaming data sets that are required to run the 
experiment notebooks and unit tests. The fully cleansed and normalized data sets can be found in ```./data/datasets```.

*[!] We will soon introduce an extensive set of benchmark data sets that can be directly accessed 
from a remote source using the float.DataLoader module.*

## References
Please refer to the following paper when using float: *Link will following soon.*

Please also make sure that you acknowledge the corresponding authors, when using one of the provided models (Relevant 
sources and papers are linked in the docstrings of each file).

## Contribution
We welcome meaningful contributions by the community. In particular, we encourage contributions of new evaluation strategies, 
performance measures, expressive visualizations and streaming data sets.

Additionally, we welcome implementations of novel online learning models. 
However, we discourage contributing models that are already included in one 
of the major libraries, e.g. [scikit-multiflow](https://scikit-multiflow.readthedocs.io/en/stable/#) or [river](https://riverml.xyz/latest/)
(i.e., float is primarily an evaluation toolkit and not a library of state-of-the-art online learning methods).

All contributed source code must adhere to the following criteria:
- Code must conform with the [PEP8](https://www.python.org/dev/peps/pep-0008/) standard.
- Docstrings must conform with the [Google docstring](https://google.github.io/styleguide/pyguide.html) convention.
- All tests in ```./unit_tests``` must run successfully. To run the complete test suite, one may execute 
```./unit_tests/run_test_suite.py```.

Please feel free to contact us, if you plan to make a contribution.
