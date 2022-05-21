import itertools
import numpy as np
import pandas as pd
from typing import List
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity


def listDiversity(predicted: list, itemsSimilarityMatrix):
    """
    Computes the diversity for a list of recommended items for a user

    Parameters
    ----------
    predicted: list
        A list of predicted numeric/character vectors of retrieved documents for the corresponding element of actual
        example: ['X', 'Y', 'Z']

    Returns
    ----------
        diversity
    """
    pairCount = 0
    similarity = 0
    pairs = itertools.combinations(predicted, 2)
    for pair in pairs:
        itemID1 = pair[0]
        itemID2 = pair[1]
        similarity += itemsSimilarityMatrix[itemID1, itemID2]
        pairCount += 1
    averageSimilarity = similarity / pairCount
    diversity = 1 - averageSimilarity
    return diversity


def novelty(predicted: list, pop: dict, u: int, k: int):
    """
    Computes the novelty for a list of recommended items for a user

    Parameters
    ----------
    predicted: a list of recommedned items
        Ordered predictions
        example: ['X', 'Y', 'Z']
    pop: dictionary
        A dictionary of all items alongside of its occurrences counter in the training data
        example: {1198: 893, 1270: 876, 593: 876, 2762: 867}
    u: integer
        The number of users in the training data
    k: integer
        The length of recommended lists per user

    Returns
    ----------
    novelty:
        The novelty of the recommendations in system level

    Metric Definition
    ----------
    Zhou, T., Kuscsik, Z., Liu, J. G., Medo, M., Wakeling, J. R., & Zhang, Y. C. (2010).
    Solving the apparent diversity-accuracy dilemma of recommender systems.
    Proceedings of the National Academy of Sciences, 107(10), 4511-4515.
    """
    selfInformation = 0
    for item in predicted:
        if item in pop.keys():
            itemPopularity = pop[item]/u
            itemNoveltyValue = np.sum(-np.log2(itemPopularity))
        else:
            itemNoveltyValue = 0
        selfInformation += itemNoveltyValue
    noveltyScore = selfInformation/k
    return noveltyScore


def catalogCoverage(predicted: List[list], catalog: set):
    """
    Computes the catalog coverage for k lists of recommendations
    Coverage is the percent of items in the training data the model is able to recommend on a test set

    Parameters
    ----------
    predicted: a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    catalog: list
        A list of all unique items in the training data
        example: ['A', 'B', 'C', 'X', 'Y', 'Z']
    k: integer
        The number of observed recommendation lists
        which randomly choosed in our offline setup

    Returns
    ----------
    catalogCoverage:
        The catalog coverage of the recommendations as a percent
        rounded to 2 decimal places

    Metric Definition
    ----------
    Ge, M., Delgado-Battenfeld, C., & Jannach, D. (2010, September).
    Beyond accuracy: evaluating recommender systems by coverage and serendipity.
    In Proceedings of the fourth ACM conference on Recommender systems (pp. 257-260). ACM.
    """
    predictedFlattened = [p for sublist in predicted for p in sublist]
    LPredictions = len(set(predictedFlattened))
    catalogCoverage = round(LPredictions / (len(catalog) * 1.0) * 100, 2)
    return catalogCoverage


def personalization(predicted: List[list]):
    """
    Personalization measures recommendation similarity across users.
    A high score indicates good personalization (user's lists of recommendations are different).
    A low score indicates poor personalization (user's lists of recommendations are very similar).
    A model is "personalizing" well if the set of recommendations for each user is different.

    Parameters
    ----------
    predicted: a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]

    Returns
    -------
        The personalization score for all recommendations.
    """

    def makeRecMatrix(predicted: List[list]):
        df = pd.DataFrame(data=predicted).reset_index().melt(
            id_vars='index', value_name='item',
        )
        df = df[['index', 'item']].pivot(
            index='index', columns='item', values='item')
        df = pd.notna(df)*1
        recMatrix = sp.csr_matrix(df.values)
        return recMatrix

    # Create matrix for recommendations
    predicted = np.array(predicted)
    recMatrixSparse = makeRecMatrix(predicted)
    # Calculate similarity for every user's recommendation list
    similarity = cosine_similarity(X=recMatrixSparse, dense_output=False)
    # Get indicies for upper right triangle w/o diagonal
    upperRight = np.triu_indices(similarity.shape[0], k=1)
    # Calculate average similarity
    personalization = np.mean(similarity[upperRight])
    return 1-personalization
