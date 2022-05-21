import numpy as np
from utils import logger
from config import USGDict
from Models.utils import loadModel, saveModel
from Models.USG.lib.FriendBasedCF import FriendBasedCF


modelName = 'USG'


def friendBasedCalculations(datasetName: str, users: dict, pois: dict, socialRelations, trainingMatrix, groundTruth):
    # Initializing parameters
    userCount = users['count']
    logDuration = 1 if userCount < 20 else 10
    eta = USGDict['eta']
    SScores = np.zeros((users['count'], pois['count']))
    # Checking for existing model
    logger('Preparing Friend-based CF matrix ...')
    loadedModel = loadModel(modelName, datasetName, f'S_{userCount}User')
    if loadedModel == []:  # It should be created
        # Creating object to S Class
        S = FriendBasedCF(eta)
        # Calculating S scores
        # TODO: We may be able to load the model from disk
        S.friendsSimilarityCalculation(socialRelations, trainingMatrix)
        for counter, uid in enumerate(users['list']):
            # Adding log to console
            if (counter % logDuration == 0):
                print(f'User#{counter} processed ...')
            if uid in groundTruth:
                for lid in pois['list']:
                    SScores[uid, lid] = S.predict(uid, lid)
                SScores = np.array(SScores)
        saveModel(SScores, modelName, datasetName, f'S_{userCount}User')
    else:  # It should be loaded
        SScores = loadedModel
    # Returning the scores
    return SScores
