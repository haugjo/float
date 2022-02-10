# Getting Started
## Installation
Install float with pip:
```
pip install float-evaluation
```
Alternatively, clone the float repository from Github:
```
git clone git@github.com:haugjo/float.git
```

### Requirements
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

## Run Your First Experiment
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
# Alternatively, we can provide a scikit-multiflow FileStream object via the 'stream' attribute.
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

To become more familiar with float and its functionality you may look at the **examples provided in this documentation**.
Specifically, we provide experiments that showcase each of the three evaluation pipelines *prequential*, *holdout* and 
*distributed fold*. We also demonstrate the use of the *change detection*, *online feature selection* and 
*visualization* modules.

## Datasets
On the float Github page, you can find a small set of popular streaming data sets that are required to run the 
example notebooks and unit tests.

Although there is a general shortage of publicly available streaming data sets, there are some useful resources:

- [https://www.openml.org/search?type=data](https://www.openml.org/search?type=data) (search for "data stream" or "concept drift")
- [https://sites.google.com/view/uspdsrepository](https://sites.google.com/view/uspdsrepository)
- [https://github.com/ogozuacik/concept-drift-datasets-scikit-multiflow](https://github.com/ogozuacik/concept-drift-datasets-scikit-multiflow)
- [https://github.com/vlosing/driftDatasets](https://github.com/vlosing/driftDatasets)