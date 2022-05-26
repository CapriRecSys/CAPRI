======================
Datasets
======================

For now, ``CAPRI`` supports three well-know datasets Point-of-Interest (POI) recommendation datasets, listed as below.
These datasets can be utilized for a wide range of tasks in the POI domain, inluding location-based recommendation, users activity recommendation, and user group recommendation.
All of the datasets are available in the `Data`_ directory of the ``CAPRI`` package.
Please note that As the framework is still in development, we are working on adding more datasets.

.. _Data: https://github.com/CapriRecSys/CAPRI/tree/main/Data

Gowalla
-------

Gowalla is a location-based social networking website where users share their locations by checking-in.
The friendship network is undirected and was collected using their public API, and consists of 196,591 nodes and 950,327 edges.
We have collected a total of 6,442,890 check-ins of these users over the period of Feb. 2009 - Oct. 2010.
Gowalla contains "Geographical", "Social", "Temporal", and "Interaction" features.
Read more at the `Stanford University website <https://snap.stanford.edu/data/loc-gowalla.html>`_

Yelp
----

The Yelp dataset is a subset of our companies, reviews, and user data that can be used for personal, scholarly, and educational reasons.
In the most recent dataset you'll find information about businesses across 8 metropolitan areas in the USA and Canada.
It satisfies "Geographical", "Social", "Temporal", "Categorical", and "Interaction" features.
Read more at the `Yelp website <https://www.yelp.com/dataset>`_


Foursquare
----------

This dataset contains check-ins in NYC and Tokyo collected for about 10 month (from 12 April 2012 to 16 February 2013).
It contains 227,428 check-ins in New York city and 573,703 check-ins in Tokyo.
Each check-in is associated with its time stamp, its GPS coordinates and its semantic meaning (represented by fine-grained venue-categories).
This dataset is originally used for studying the spatial-temporal regularity of user activity in LBSNs.
It satisfies "Geographical", "Social", "Temporal", and "Interaction" features.
Read more at the `Kaggle website <https://www.kaggle.com/datasets/chetanism/foursquare-nyc-and-tokyo-checkin-dataset>`_

