import numpy as np
from utils import logger
from config import LoreDict
from Models.utils import loadModel, saveModel
from Models.LORE.lib.AdditiveMarkovChain import AdditiveMarkovChain


modelName = 'LORE'


def additiveMarkovChainCalculations(datasetName: str, users: dict, pois: dict, sortedTrainingCheckins, groundTruth):
    """
    This function calculates the additive markov chain features of the dataset.

    Parameters
    ----------
    datasetName : str
        The name of the dataset
    users : dict
        The users of the dataset
    pois : dict
        The pois of the dataset
    sortedTrainingCheckins : dict
        The sorted training checkins of the dataset
    groundTruth : dict
        The ground truth of the dataset

    Returns
    -------
    KDEScores : dict
        The KDE scores of the dataset
    """
    # Initializing parameters
    userCount = users['count']
    logDuration = 1 if userCount < 20 else 10
    alpha, deltaT = LoreDict['alpha'], LoreDict['deltaT']
    AMCScores = np.zeros((users['count'], pois['count']))
    # Checking for existing model
    logger('Preparing Additive Markov Chain matrix ...')
    loadedModel = loadModel(modelName, datasetName, f'AMC_{userCount}User')
    if loadedModel == []:  # It should be created
        # Creating object to AMC Class
        AMC = AdditiveMarkovChain(deltaT, alpha)
        # Calculating AMC scores
        # TODO: We may be able to load the model from disk
        AMC.buildLocationToLocationTransitionGraph(sortedTrainingCheckins)
        for counter, uid in enumerate(users['list']):
            # Adding log to console
            if (counter % logDuration == 0):
                print(f'User#{counter} processed ...')
            if uid in groundTruth:
                for lid in pois['list']:
                    AMCScores[uid, lid] = AMC.predict(uid, lid)
        saveModel(AMCScores, modelName, datasetName,
                  f'AMC_{userCount}User')
    else:  # It should be loaded
        AMCScores = loadedModel
    # Returning the scores
    return AMCScores
