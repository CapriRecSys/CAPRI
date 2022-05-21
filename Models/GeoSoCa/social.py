import numpy as np
from utils import logger
from Models.utils import loadModel, saveModel
from Models.GeoSoCa.lib.SocialCorrelation import SocialCorrelation

modelName = 'GeoSoCa'


def socialCalculations(datasetName: str, users: dict, pois: dict, trainingMatrix, socialRelations, groundTruth):
    """
    This function is used to calculate the social correlation between users and pois.

    Parameters
    ----------
    datasetName : str
        The name of the dataset.
    users : dict
        The dictionary of users.
    pois : dict
        The dictionary of pois.
    trainingMatrix : np.array
        The training matrix of the dataset.
    socialRelations : dict
        The dictionary containing the social relations of the dataset.
    groundTruth : dict
        The dictionary containing the ground truth of the dataset.

    Returns
    -------
    socialCorrelation : dict
        The dictionary containing the social correlation of the dataset.
    """
    # Initializing parameters
    userCount = users['count']
    logDuration = 1 if userCount < 20 else 10
    SCScores = np.zeros((userCount, pois['count']))
    # Checking for existing model
    logger('Preparing Social Correlation matrix ...')
    loadedModel = loadModel(modelName, datasetName,
                            f'SC_{userCount}User')
    if loadedModel == []:  # It should be created
        # Creating object to AKDE Class
        SC = SocialCorrelation()
        # Social Correlation Calculations
        loadNumpyArray = loadModel(modelName, datasetName, 'Beta')
        if loadNumpyArray == []:  # It should be created
            SC.computeBeta(trainingMatrix, socialRelations)
            saveModel(SC.X, modelName, datasetName, 'Beta')
        else:  # It should be loaded
            SC.loadModel(loadNumpyArray)
        # Calculating SC scores
        print("Now, training the model for each user ...")
        for counter, uid in enumerate(users['list']):
            # Adding log to console
            if (counter % logDuration == 0):
                print(f'User#{counter} processed ...')
            if uid in groundTruth:
                for lid in pois['list']:
                    SCScores[uid, lid] = SC.predict(uid, lid)
        saveModel(SCScores, modelName, datasetName,
                  f'SC_{userCount}User')
    else:  # It should be loaded
        SCScores = loadedModel
    # Returning the scores
    return SCScores
