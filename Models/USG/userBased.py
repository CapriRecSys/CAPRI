import numpy as np
from utils import logger
from Models.utils import loadModel, saveModel
from Models.USG.lib.UserBasedCF import UserBasedCF

modelName = 'USG'


def userBasedCalculations(datasetName: str, users: dict, pois: dict, trainingMatrix, groundTruth):
    # Initializing parameters
    userCount = users['count']
    logDuration = 1 if userCount < 20 else 10
    UScores = np.zeros((users['count'], pois['count']))
    # Checking for existing model
    logger('Preparing User-based CF matrix ...')
    loadedModel = loadModel(modelName, datasetName, f'U_{userCount}User')
    if loadedModel == []:  # It should be created
        # Creating object to U Class
        U = UserBasedCF()
        # User-based Collaborative Filtering Calculations
        numpyArray = loadModel(modelName, datasetName, 'recScore')
        if numpyArray == []:  # It should be created
            U.preComputeRecScores(trainingMatrix)
            saveModel(U.recScore, modelName, datasetName, 'recScore')
        else:  # It should be loaded
            U.loadModel(numpyArray)
        for counter, uid in enumerate(users['list']):
            # Adding log to console
            if (counter % logDuration == 0):
                print(f'User#{counter} processed ...')
            if uid in groundTruth:
                for lid in pois['list']:
                    UScores[uid, lid] = U.predict(uid, lid)
                UScores = np.array(UScores)
        saveModel(UScores, modelName, datasetName, f'U_{userCount}User')
    else:  # It should be loaded
        UScores = loadedModel
    # Returning the scores
    return UScores
