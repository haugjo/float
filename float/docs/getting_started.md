# Getting Started
## Installation
Install float with pip:
```
pip install float
```
Alternatively, clone the float repository from Github:
```
git clone git@github.com:haugjo/float.git
```

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

## Simple Experiment
The easiest way to become familiar with float and its functionality is to look at the experiment notebooks in the ```./float/experiments``` 
folder. We provide experiments that showcase each of the three evaluation pipelines *prequential*, *holdout* and 
*distributed fold*. We also demonstrate the use of the *change detection*, *online feature selection* and 
*visualization* modules. 

*[!]Note that each module and file of float is fully documented via docstring. A dedicated documentation page is soon 
to follow. More information about the general package structure is provided in the section below.*

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
data_loader = DataLoader(path='./float/data/datasets/spambase.csv', target_col=-1)

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

## Datasets
Float is accompanied by a small set of popular streaming data sets that are required to run the 
experiment notebooks and unit tests. These fully cleansed and normalized data sets can be found in ```./data/datasets```.

There is a general shortage of publicly available streaming data sets. Still, there are useful resources, including the following:

- https://www.openml.org/search?type=data (search for "data stream" or "concept drift")
- https://sites.google.com/view/uspdsrepository
- https://github.com/ogozuacik/concept-drift-datasets-scikit-multiflow
- https://github.com/vlosing/driftDatasets

*[!] We will soon introduce an extensive set of benchmark data sets that can be directly accessed 
from a remote source using the float.DataLoader module.*