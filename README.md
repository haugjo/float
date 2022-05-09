<p align="center">
  <img alt="float" src="https://github.com/haugjo/float/raw/main/logo.png" width="400"/>
</p>
<p align="center">
  <!-- Pypi -->
  <a href="https://pypi.org/project/float-evaluation/">
    <img src="https://img.shields.io/pypi/v/float-evaluation" alt="Pypi">
  </a>
  <!-- Documentation -->
  <a href="https://haugjo.github.io/float/">
    <img src="https://img.shields.io/badge/docs-pages-informational" alt="Documentation">
  </a>
  <!-- License -->
  <a href="https://github.com/haugjo/float/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/haugjo/float" alt="GitHub license">
  </a>
</p>

<p align="center">
    <strong><em>float</em> (FrictionLess Online model Analysis and Testing)</strong> is a modular Python-framework for standardized evaluations of online learning methods.
    <em>float</em> can combine custom code with popular libraries and provides easy access to sensible evaluation strategies, measures and visualisations.
</p>

>**Release Notes:**<br>
>*02/02/22* - Initial release.

## Installation
Install float with pip:
```
pip install float-evaluation
```
Alternatively, execute the following code in a command line to clone the float repository from Github:
```
git clone git@github.com:haugjo/float.git
```

## Requirements
Float is supported by python versions >=3.8.x and has the following package requirements:
- numpy >= 1.22.1
- scikit-learn >= 1.0.2
- tabulate >= 0.8.5
- matplotlib >= 3.4.2
- scipy >= 1.7.3
- pandas >= 1.4.0
- river >= 0.9.0
- scikit-multiflow >= 0.5.3

*[!] Older versions have not been tested, but might also be compatible.*

## Quickstart
The easiest way to become familiar with float and its functionality is to look at the example notebooks in the documentation or ```./docs/examples``` respectively. 
We provide experiments that showcase each of the three evaluation pipelines *prequential*, *holdout* and 
*distributed fold*. We also demonstrate the use of the *change detection*, *online feature selection* and 
*visualization* modules.

As a quickstart, you can run the following basic experiment:
```python
from skmultiflow.trees import HoeffdingTreeClassifier
from sklearn.metrics import zero_one_loss

from float.data import DataLoader
from float.prediction.evaluation import PredictionEvaluator
from float.prediction.evaluation.measures import noise_variability
from float.pipeline import PrequentialPipeline
from float.prediction.skmultiflow import SkmultiflowClassifier

# Load a data set from main memory with the DataLoader module. 
# Alternatively, we can provide a sciki-multiflow FileStream object via the 'stream' attribute.
data_loader = DataLoader(path='./datasets/spambase.csv', target_col=-1)

# Set up an online classifier. Note that we need a wrapper to use scikit-multiflow functionality.
classifier = SkmultiflowClassifier(model=HoeffdingTreeClassifier(),
                                   classes=data_loader.stream.target_values)

# Set up an evaluator object for the classifier:
# Specifically, we want to measure the zero_one_loss and the noise_variability as an indication of robustness.
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
## Package Structure
Float is fully modular. The package has dedicated modules for the data preparation, online prediction, online 
feature selection, concept drift detection and their evaluation (corresponding to the directories in ```./float```). 

The pipeline module is at the heart of every experiment in float. The pipeline can be run with arbitrary combinations of 
predictors, online feature selectors and concept drift detectors. 

The results of every experiment are stored in dedicated evaluator objects. Via these evaluators, the user is 
able to specify the performance measures that will be computed during the evaluation. After running the pipeline, the 
evaluators can be used to create custom tables and plots or to obtain standardized visualizations with the float visualization module. 

Each of the above-mentioned modules contains an abstract base class that simplifies the integration of custom 
functionality in the experiment. 

Float also provides wrappers for [scikit-mukltiflow](https://scikit-multiflow.readthedocs.io/en/stable/index.html) 
and [river](https://riverml.xyz/latest/) functionality wherever available. Besides float integrates the change detectors
implemented by the [Tornado](https://github.com/alipsgh/tornado) package.

## Datasets
Float is accompanied by a small set of popular streaming data sets that are required to run the 
experiment notebooks and unit tests. These fully cleansed and normalized data sets can be found in ```./datasets```.

Although there is a general shortage of publicly available streaming data sets, there are some useful resources:
- https://www.openml.org/search?type=data (search for "data stream" or "concept drift")
- https://sites.google.com/view/uspdsrepository
- https://github.com/ogozuacik/concept-drift-datasets-scikit-multiflow
- https://github.com/vlosing/driftDatasets

## References
Please refer to the following paper when using float:
*Haug, Johannes, Effi Tramountani, and Gjergji Kasneci. "Standardized Evaluation of Machine Learning Methods for Evolving Data Streams." arXiv preprint [arXiv:2204.13625](https://arxiv.org/abs/2204.13625) (2022).*

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
- All unit tests in ```./tests``` must run successfully. To run the complete test suite, one may execute 
```./tests/run_test_suite.py```.

Please feel free to contact us, if you plan to make a contribution.
