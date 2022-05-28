======
Contribution
======


There are many ways to contribute to ``CAPRI``!
You can contribute code, make improvements to the documentation, report or investigate `bugs and issues <https://github.com/CARecSys/CAPRI/issues/>`_.
We welcome all contributions from bug fixes to new features and extensions.
Feel free to share with us your custom configuration files.
We are creating a vault of reproducible experiments, and we would be glad of mentioning your contribution.
Reference ``CAPRI`` in your blogs, papers, and articles.
Also, talk about CAPRI on social media with the hashtag ``#capri_framework``.
Contribution to the project can be done through various approaches:


Adding a new dataset
---------

All datasets can be found in `Data`_ directory. In order to add a new dataset, you should:

.. _Data: https://github.com/CapriRecSys/CAPRI/tree/main/Data

Modify the ``config.py`` file and add a record to the datasets dictionary.
The key of the item should be the dataset's name (CapitalCase) and the value is an array of strings containing the dataset scopes (all CapitalCase).
For instance:

::


    "DatasetName":  ["Scope1", "Scope2", "Scope3"]


Add a folder to the `Data`_ directory with the exact same name selected in the previous step.
This way, your configs are attached to the dataset.
In the created folder, add files of the dataset (preferably camelCase, e.g. socialRelations).
Note that for each of these files, a variable with the exact same name will be automatically generated and fed to the models section.
You can find a sample for the dataset sturcture here:

.. _Data: https://github.com/CapriRecSys/CAPRI/tree/main/Data

::


    + Data/
        + Dataset1
            + datasetFile1
            + datasetFile2
            + datasetFile3
        + Dataset2
            + datasetFile4
            + datasetFile5
            + datasetFile6


Adding a new model
---------

Models can be found in `Models`_ directory. In order to add a new model, you should:

.. _Models: https://github.com/CapriRecSys/CAPRI/tree/main/Models

Modify the ``config.py`` file and add a record to the models dictionary.
The key of the item should be the model's name (CapitalCase) and the value is an array of strings containing the scopes that mode covers (all CapitalCase).
For instance:

::

    "ModelName":  ["Scope1", "Scope2", "Scope3"]


Add a folder to the `Models`_ directory with the exact same name selected in the previous step.
This way, your configs are attached to the model.
In the created folder, add files of the model (preferably camelCase, e.g. socialRelations).
Models contain a main.py file that holds the contents of the model.
The file main.py contains a class with the exact name of the model and the letter 'Main' (e.g. ModelNameMain).
This class should contain a main function with two argument: (i) datasetFiles dictionary, (ii) the parameters of the selected model (including top-K items for evaluation, sparsity ratio, restricted list for computation, and dataset name).
For a better description, check the code sample below:

.. _Models: https://github.com/CapriRecSys/CAPRI/tree/main/Models

::

    import numpy as np
    ...

    class NewModelMain:
        def main(datasetFiles, parameters):
            print('Other codes goes here')

There is a ``utils.py`` file in the `Models`_ directory that keeps the utilities that can be used in all models.
If you are thinking about a customized utilities with other functions, you can add an ``extendedUtils.py`` file in the model's directory.
Also, a ``/lib/`` directory is considered in each model folders that contains the libraries used in the model.
You can find a sample for the dataset sturcture here:

.. _Models: https://github.com/CapriRecSys/CAPRI/tree/main/Models

::

    + Models/
        + Model1/
            + lib/
            + __init__.py
            + main.py
            + extendedUtils.py
        + utils.py
        + __init__.py

Note: do not forget to add a ``init.py`` file to the directories you make.


Adding a new evaluation metric
---------

You can simply add the evaluations to the `accuracy.py`_ or `beyoundAccuracy.py`_ files.
Please note that to ensure the evaluation modules work correctly, we use the Python unit test library which can be found in `test.py`_.
To find out more about how unit tests work, check this `Digital-Ocean link <https://jingwen-z.github.io/how-to-apply-mock-with-python-unittest-module/>`_ .
Hereby, please always consider adding a unit test for the evaluation modules you add.
To run the test module, you can run the following command:

.. _accuracy.py: https://github.com/CapriRecSys/CAPRI/blob/main/Evaluations/metrics/accuracy.py
.. _beyoundAccuracy.py: https://github.com/CapriRecSys/CAPRI/blob/main/Evaluations/metrics/beyoundAccuracy.py
.. _test.py: https://github.com/CapriRecSys/CAPRI/blob/main/Evaluations/test.py

::

    python -m unittest test.py