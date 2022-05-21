import numpy as np


def precisionk(actual: list, recommended: list):
    """
    Computes the number of relevant results among the top k recommended items

    Parameters
    ----------
    actual: list
        A list of ground truth items
        example: [X, Y, Z]
    recommended: list
        A list of ground truth items (all possible relevant items)
        example: [x, y, z]

    Returns
    ----------
        precision at k
    """
    relevantResults = set(actual) & set(recommended)
    assert 0 <= len(
        relevantResults), f"The number of relevant results is not true (currently: {len(relevantResults)})"
    return 1.0 * len(relevantResults) / len(recommended)


def recallk(actual: list, recommended: list):
    """
    The number of relevant results among the top k recommended items divided by the total number of relevant items

    Parameters
    ----------
    actual: list
        A list of ground truth items (all possible relevant items)
        example: [X, Y, Z]
    recommended: list
        A list of items recommended by the system
        example: [x, y, z]

    Returns
    ----------
        recall at k
    """
    relevantResults = set(actual) & set(recommended)
    assert 0 <= len(
        relevantResults), f"The number of relevant results is not true (currently: {len(relevantResults)})"
    return 1.0 * len(relevantResults) / len(actual)


def mapk(actual: list, predicted: list, k: int = 10):
    """
    Computes mean Average Precision at k (mAPk) for a set of recommended items

    Parameters
    ----------
    actual: list
        A list of ground truth items (order doesn't matter)
        example: [X, Y, Z]
    predicted: list
        A list of predicted items, recommended by the system (order matters)
        example: [x, y, z]
    k: integer, optional (default to 10)
        The number of elements of predicted to consider in the calculation

    Returns
    ----------
    score:
        The mean Average Precision at k (mAPk)
    """
    score = 0.0
    numberOfHits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            numberOfHits += 1.0
            score += numberOfHits / (i+1.0)
    if not actual:
        return 0.0
    score = score / min(len(actual), k)
    return score


def dcg(scores: list):
    """
    Computes the Discounted Cumulative Gain (DCG) for a list of scores

    Parameters
    ----------
    scores: list
        A list of scores

    Returns
    ----------
    dcg: float
        The Discounted Cumulative Gain (DCG)
    """
    return np.sum(np.divide(np.power(2, scores) - 1, np.log(np.arange(scores.shape[0], dtype=np.float32) + 2)),
                  dtype=np.float32)


def ndcgk(actual: list, predicted: list, relevance=None, at=None):
    """
    Calculates the implicit version of Normalized Discounted Cumulative Gain (NDCG) for top k items in the ranked output

    Parameters
    ----------
    actual: list
        A list of ground truth items
        example: [X, Y, Z]
    predicted: list
        A list of predicted items, recommended by the system
        example: [x, y, z]
    relevance: list, optional (default to None)
        A list of relevance scores for the items in the ground truth
    at: any, optional (default to None)
        The number of items to consider in the calculation

    Returns
    ----------
    ndcg:
        Normalized DCG score

    Metric Defintion
    ----------
    Jarvelin, K., & Kekalainen, J. (2002). Cumulated gain-based evaluation of IR techniques.
    ACM Transactions on Information Systems (TOIS), 20(4), 422-446.
    """
    # Convert list to numpy array
    actual, predicted = np.asarray(list(actual)), np.asarray(list(predicted))
    # Check the relevance value
    if relevance is None:
        relevance = np.ones_like(actual)
    assert len(relevance) == actual.shape[0]
    # Creating a dictionary associating itemId to its relevance
    item2rel = {it: r for it, r in zip(actual, relevance)}
    # Creates array of length "at" with the relevance associated to the item in that position
    rankScores = np.asarray([item2rel.get(it, 0.0)
                            for it in predicted[:at]], dtype=np.float32)
    # IDCG has all relevances to 1, up to the number of items in the test set
    idcg = dcg(np.sort(relevance)[::-1])
    # Calculating rank-DCG, DCG uses the relevance of the recommended items
    rdcg = dcg(rankScores)
    if rdcg == 0.0:
        return 0.0
    # Return items
    return round(rdcg / idcg, 4)
