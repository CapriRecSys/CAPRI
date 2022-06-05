======
Configurations
======

``CAPRI`` contains a ``config.py`` file that provides adjustments and configurations for running experiments.
One of the benefits of CAPRI is that the outcomes of the previously ran experiments and recommendation computations will be store in a defined directory.
Thus, users can always follow-up and explore additional properties of the results.
The parameters that can be set using this file includes:

dataDirectory
--------------

It contains the path to read data from.

outputsDir
----------

It holds the path to store recommendation lists.

topK
---------

Refers to Top-k items for evaluation (default: 10).

limitUsers
-----------

Limiting the number of users in dataset (default: -1)

listLimit
---------

Limiting the length of recommendation lists (default: 10)

activeUsersPercentage
----------------------

Calculating a list of defined percentages of users as "active users"

models
---------

List of available models to be selected by users.

datasets
---------

List of available datasets to be selected by users

fusions
---------

List of available fusions to be selected by users

evaluationMetrics
------------------

List of available evaluation metrics to be selected by users
