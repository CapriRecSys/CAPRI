======================
Evaluation Metrics
======================


Many evaluation metrics are available for recommendation systems and each has its own pros and cons.
``CAPRI`` supports the two types of evaluation metrics: **Accuracy** and **Beyond Accuracy**.
In this page, we will discuss these two evaluation metrics in detail.


Accuracy Metrics
--------------------

Precision@K
~~~~~~~~~~~~~~~~

Precision at k is the fraction of relevant recommended items in the top-k set.
Assume that in a top-10 recommendation problem, my precision at 10 is 75%.
This means that 75% of the suggestions I offer are applicable to the user.

::

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


Recall@K
~~~~~~~~~~~~~~~~

The proportion of relevant things discovered in the top-k suggestions is known as recall at k.
Assume we computed recall at 10 and discovered it to be 55% in our top-10 recommendation system.
This suggests that the top-k results contain 55% of the entire number of relevant items.

::

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
        assert 0 <= len(relevantResults), f"The number of relevant results is not true (currently: {len(relevantResults)})"
        return 1.0 * len(relevantResults) / len(actual)


Map@K
~~~~~~~~~~~~~~~~

MAP@k stands for Mean Average Precision at Cut Off k and is most commonly employed in recommendation systems, 
although it can also be utilized in other systems.
When you use MAP to evaluate a recommender algorithm, you're treating the recommendation as a ranking problem.

::

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


NDCG@K
~~~~~~~~~~~~~~~~

The Discounted Cumulative Gain for k displayed recommendations sums up the importance of the items displayed for 
the current user (cumulative), while penalizing relevant items in later slots (discounted).
In the Normalized Cumulative Gain for k given suggestions, this score is divided by the maximum possible value of DCG@K for the current user.

::

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
    return np.sum(np.divide(np.power(2, scores) - 1, np.log(np.arange(scores.shape[0], dtype=np.float32) + 2)), dtype=np.float32)


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


Beyond-accuracy Metrics
--------------------------------------

Beyound accuracy metrics refer to evaluating recommender systems by coverage and serendipity.


List Diversity
~~~~~~~~~~~~~~~~

List Diversity is one of the most common metrics used to evaluate recommender systems.

::

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

Novelty
~~~~~~~~~~~~~~~~

Novelty is one of the most common metrics used to evaluate recommender systems.

::

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


Catalog Coverage
~~~~~~~~~~~~~~~~

Catalog Coverage is one of the most common metrics used to evaluate recommender systems.

::

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

Personalization
~~~~~~~~~~~~~~~~

Personalization is one of the most common metrics used to evaluate recommender systems.

::

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