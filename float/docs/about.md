# About
## Vision & Scope
Todo

## References
Please refer to the following paper when using float: *Link will following soon.*

Please also make sure that you acknowledge the corresponding authors, when using one of the provided models (Relevant 
sources and papers are linked in the docstrings of each file).
 
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