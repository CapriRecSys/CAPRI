=============================
CAPRI, A POI RecSys Framework
=============================


.. .. image:: https://coveralls.io/repos/github/that-recsys-lab/librec-auto/badge.svg?branch=master
..   :target: https://coveralls.io/github/that-recsys-lab/librec-auto?branch=master

About
=====

``CAPRI`` stands for "Context-Aware Interpretable Point-of-Interest Recommendation Framework".
It is a Python-based tool for standardizing algorithms' implementations and enabling the reproducibility of experiments
in the field of Point-of-Interest (POI) recommendation. In ``CAPRI``, we provided a set of datasets, models, and algorithms
to simplify the process of reproducibility and enable the comparison of algorithms' performance.

.. _CAPRI: https://github.com/CapriRecSys/CAPRI

``CAPRI`` covers state-of-the-art models and algorithms and well-known datasets in the field of POI recommendations.
As it provides and supports reproducibility of results, users can easily share their experimental settings to
achieve the same outcomes for comparative experiments.
This contains pre-implemented methodological workflows, a wide range of models and algorithms, and
various datasets for benchmarking purposes. Other features of ``CAPRI`` include:

* Producing recommendation lists based on a wide range of configurations
* Configuration using a wide range of choices
* Ability to run experiments using various evaluation metrics

Workflow
========

The framework is released under GPL v3 License and can be accessed (download or clone) via GitHub.


Configuration
=============

TBA

Repo Structure
=================

Regarding the implementation, the files of the frameworks are organized in multiple directories for easier
accessibility and extensibility purposes.
As a general naming structure and to combine words into a single string in CAPRI, we use PascalCase and camelCase
for folder and file names, respectively. Below, the main directories of the framework containing files are presented in brief:

* `Data`_: Contains all sorts of data-driven files and functions. Each dataset contains different files with ``.txt`` extensions, including train, test, and tune data. Moreover, other files containing the check-ins data and relations among users/items such as social and geographical data are stored in folders with the same name as each dataset.

* `Models`_: Contains the models used in the framework and some common functions in the ``utils.py`` file to avoid code repetition and improve re-usability of model files.

* `Evaluations`_: Contains all evaluation metrics available to analyse the performance of models on datasets, the evaluator function ``evaluator.py`` that utilizes the mentioned metrics, and a unit test file ``test.py`` for checking the performance of each metric with different input types.

* `Outputs`_: Keeps the final results, including ranked lists and evaluation outputs produced by the framework.

.. _Data: https://github.com/CapriRecSys/CAPRI/tree/main/Data
.. _Models: https://github.com/CapriRecSys/CAPRI/tree/main/Models
.. _Evaluations: https://github.com/CapriRecSys/CAPRI/tree/main/Evaluations
.. _Outputs: https://github.com/CapriRecSys/CAPRI/tree/main/Outputs