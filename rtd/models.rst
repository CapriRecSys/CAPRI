======
Models
======


``CAPRI`` contains various models for analysis of the data in Point-of-Interest (POI) datasets.
All of the models are available in the `Models`_ directory of the ``CAPRI`` package.
Please note that As the framework is still in development, we are working on adding more models.

.. _Models: https://github.com/CapriRecSys/CAPRI/tree/main/Models


Baselines
---------

TBD

Context-aware POI Recommendation
--------------------------------

Context Aware Recommendation Systems incorporate a variety of contextual factors in order to accurately capture user preferences.

GeoSoCa
~~~~~~~~~~~~~~~~

GoeSoCa is a novel POI recommendation method that uses geographical, social, and category correlations between users and POIs to make recommendations.
These correlations can be learned from user check-in data on POIs in the past and used to predict a user's relevance score to an unvisited POI in order to offer suggestions to them.
Read more at `GeoSoCa's paper <https://dl.acm.org/doi/10.1145/2766462.2767711>`_
You can also check the content of `GeoSoCa`_ model in ``CAPRI`` package.
.. _GeoSoCa: https://github.com/CapriRecSys/CAPRI/tree/main/Models/GeoSoCa


LORE
~~~~~~~~~~~~~~~~

LORE is another model utilized in the context of context-aware POI recommendation systems.
It is a popular and robust model for location recommendation focused on the impacts of geographical and social influence on users’ check-in behaviors.
LORE incrementally mines sequential patterns from location sequences and represents the sequential patterns as a dynamic Location-Location Transition Graph (L2TG).
It also predicts the probability of a user visiting a location by Additive Markov Chain (AMC) with L2TG.
Finally, it fuses sequential influence with geographical influence and social influence into a unified recommendation framework.
Read more at `LORE's paper <https://dl.acm.org/doi/10.1145/2666310.2666400>`_
You can also check the content of `LORE`_ model in ``CAPRI`` package.
.. _LORE: https://github.com/CapriRecSys/CAPRI/tree/main/Models/LORE

USG
~~~~~~~~~~~~~~~~

USG is a well-known model in the POI recommedner community.
Due to the spatial clustering phenomenon demonstrated in LBSN user check-in activities, USG places a specific emphasis on geographical impact in addition to deriving user preference based on researching social influence from peers.
Accordingly, geographical influence among POIs has a significant impact on user check-in behaviors, which is modedl in USG using a power law distribution.
This model creates a naïve Bayesian-based collaborative recommendation system based on geographical influence.
Read more at `USG's paper <https://dl.acm.org/doi/10.1145/2009916.2009962>`_
You can also check the content of `USG`_ model in ``CAPRI`` package.
.. _USG: https://github.com/CapriRecSys/CAPRI/tree/main/Models/USG