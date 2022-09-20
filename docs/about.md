# About
## Vision & Scope
Float is a modular framework for simple and more standardized evaluations of online learning methods. Our framework provides easy and high-level access to popular evaluation strategies and measures, as described above. In this way, float handles large parts of the evaluation and reduces the possibility of human error.

Float enables joint integration of popular Python libraries and custom functionality. Accordingly, float is a meaningful extension of comprehensive libraries like [scikit-mukltiflow](https://scikit-multiflow.readthedocs.io/en/stable/index.html) or [river](https://riverml.xyz/latest/). In this sense, float is not intended to be another library of state-of-the-art models. Rather, our goal is to provide tools for creating high-quality experiments and visualisations.

## References
Please refer to the following paper when using float:
*Haug, Johannes, Effi Tramountani, and Gjergji Kasneci. "Standardized Evaluation of Machine Learning Methods for Evolving Data Streams." arXiv preprint [arXiv:2204.13625](https://arxiv.org/abs/2204.13625) (2022).*

Please also make sure that you acknowledge the corresponding authors, when using one of the provided models (Relevant 
sources and papers are linked in the documentation of each module).
 
## Package Structure
Float is fully modular. The package has dedicated modules for the data preparation, online prediction, online 
feature selection, concept drift detection and their evaluation. 

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
- All unit tests in ```./tests``` (see Github) must run successfully. To run the complete test suite, one may execute 
```./tests/run_test_suite.py```.

Please feel free to contact us, if you plan to make a contribution.