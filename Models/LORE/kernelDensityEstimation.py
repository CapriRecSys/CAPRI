import numpy as np
from utils import logger
from Models.utils import loadModel, saveModel
from Models.LORE.lib.KernelDensityEstimation import KernelDensityEstimation


modelName = 'LORE'


def kernelDensityEstimationCalculations(datasetName: str, users: dict, pois: dict, poiCoos, sparseTrainingMatrix, groundTruth):
    """
    This function calculates the kernel density estimation features of the dataset.

    Parameters
    ----------
    datasetName : str
        The name of the dataset
    users : dict
        The users of the dataset
    pois : dict
        The pois of the dataset
    poiCoos : dict
        The poi coordinates of the dataset
    sparseTrainingMatrix : dict
        The sparse training matrix of the dataset
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
    KDEScores = np.zeros((users['count'], pois['count']))
    # Checking for existing model
    logger('Preparing Kernel Density Estimation matrix ...')
    loadedModel = loadModel(modelName, datasetName, f'KDE_{userCount}User')
    if loadedModel == []:  # It should be created
        # Creating object to KDE Class
        KDE = KernelDensityEstimation()
        # Calculating KDE scores
        # TODO: We may be able to load the model from disk
        KDE.precomputeKernelParameters(sparseTrainingMatrix, poiCoos)
        for counter, uid in enumerate(users['list']):
            # Adding log to console
            if (counter % logDuration == 0):
                print(f'User#{counter} processed ...')
            if uid in groundTruth:
                for lid in pois['list']:
                    KDEScores[uid, lid] = KDE.predict(uid, lid)
        saveModel(KDEScores, modelName, datasetName,
                  f'KDE_{userCount}User')
    else:  # It should be loaded
        KDEScores = loadedModel
    # Returning the scores
    return KDEScores
