import numpy as np
from utils import logger
from Models.utils import loadModel, saveModel
from Models.GeoSoCa.lib.CategoricalCorrelation import CategoricalCorrelation

modelName = 'GeoSoCa'


def categoricalCalculations(datasetName: str, users: dict, pois: dict, trainingMatrix, poiCategoryMatrix, groundTruth):
    """
    This function is used to calculate the categorical features of the dataset.

    Parameters
    ----------
    datasetName : str
        The name of the dataset.
    users : dict
        The dictionary containing the users of the dataset.
    pois : dict
        The dictionary containing the pois of the dataset.
    trainingMatrix : numpy.ndarray
        The training matrix of the dataset.
    poiCategoryMatrix : numpy.ndarray
        The poi category matrix of the dataset.
    groundTruth : dict
        The ground truth of the dataset.

    Returns
    -------
    CCScores : dict
        The dictionary containing the categorical features of the dataset.
    """
    # Initializing parameters
    userCount = users['count']
    logDuration = 1 if userCount < 20 else 10
    CCScores = np.zeros((userCount, pois['count']))
    # Checking for existing model
    logger('Preparing Categorical Correlation matrix ...')
    loadedModel = loadModel(modelName, datasetName,
                            f'CC_{userCount}User')
    if loadedModel == []:  # It should be created
        # Creating object to AKDE Class
        CC = CategoricalCorrelation()
        # Category Correlation Calculations
        loadNumpyArray = loadModel(modelName, datasetName, 'Gamma')
        if loadNumpyArray == []:  # It should be created
            CC.computeGamma(trainingMatrix, poiCategoryMatrix)
            saveModel(CC.Y, modelName, datasetName, 'Gamma')
        else:  # It should be loaded
            CC.loadModel(loadNumpyArray)
        # Computing the final scores
        print("Now, training the model for each user ...")
        for counter, uid in enumerate(users['list']):
            # Adding log to console
            if (counter % logDuration == 0):
                print(f'User#{counter} processed ...')
            if uid in groundTruth:
                for lid in pois['list']:
                    CCScores[uid, lid] = CC.predict(uid, lid)
        saveModel(CCScores, modelName, datasetName,
                  f'CC_{userCount}User')
    else:  # It should be loaded
        CCScores = loadedModel
    # Returning the scores
    return CCScores
