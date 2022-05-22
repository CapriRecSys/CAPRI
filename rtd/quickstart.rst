================
Quickstart guide
================

Installation
============

You can easily clone the GitHub repository of ``CAPRI`` as below:

::
	$ git clone https://github.com/CapriRecSys/CAPRI.git

Then, install ``CAPRI`` 's required libraries using the command below:

::
	$ pip install -r requirements.txt


Storing the Results
====================

The final results of running the experiments, including a file containing the list of recommendation and a file containing the evaluation results will be stored on the ``Outputs`` directory.

::

	root
	└── Outputs

You can see that the names of the stored files starts with ``Eval_`` or ``Rec_`` prefix, which indicates the "Evaluation Results" and "Recommendation Lists," respectively.
For instance, ``Eval_GeoSoCa_Gowalla_Sum_5628user_top10_limit15.txt`` indicating Evaluation results of running GeoSoCa on Gowalla dataset using Sum function, applied on 5628 users, with selected top 10 results for evaluation and list size of 15.

Also, as running experiments takes a lot of time, the framework automatically stores calculations in shape of Python Numpy arrays in the ``Models`` directory:

::

	└── Models
		└── GeoSoCa
			└── savedModels
		└── LORE
			└── savedModels
		└── USG
			└── savedModels
