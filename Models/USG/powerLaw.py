import numpy as np
from utils import logger
from Models.USG.lib.PowerLaw import PowerLaw
from Models.utils import loadModel, saveModel

modelName = 'USG'


def powerLawCalculations(datasetName: str, users: dict, pois: dict, trainingMatrix, poiCoos, groundTruth):
    # Initializing parameters
    userCount = users['count']
    logDuration = 1 if userCount < 20 else 10
    GScores = np.zeros((users['count'], pois['count']))
    # Checking for existing model
    logger('Preparing Power Law matrix ...')
    loadedModel = loadModel(modelName, datasetName, f'G_{userCount}User')
    if loadedModel == []:  # It should be created
        # Creating object to G Class
        G = PowerLaw()
        # Calculating G scores
        # TODO: We may be able to load the model from disk
        G.fitDistanceDistribution(trainingMatrix, poiCoos)
        for counter, uid in enumerate(users['list']):
            # Adding log to console
            if (counter % logDuration == 0):
                print(f'User#{counter} processed ...')
            if uid in groundTruth:
                for lid in pois['list']:
                    GScores[uid, lid] = G.predict(uid, lid)
                GScores = np.array(GScores)
        saveModel(GScores, modelName, datasetName, f'G_{userCount}User')
    else:  # It should be loaded
        GScores = loadedModel
    # Returning the scores
    return GScores
