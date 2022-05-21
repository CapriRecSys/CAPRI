import numpy as np
from utils import logger
from config import limitUsers
from Evaluations.evaluator import evaluator
from Data.readDataSizes import readDataSizes
from Data.calculateActiveUsers import calculateActiveUsers
from Models.LORE.friendBased import friendBasedCalculations
from Models.LORE.additiveMarkovChain import additiveMarkovChainCalculations
from Models.LORE.kernelDensityEstimation import kernelDensityEstimationCalculations
from Models.utils import readFriendData, readPoiCoos, readSparseTrainingData, readTestData, readTrainingCheckins

modelName = 'LORE'


class LOREMain:
    def main(datasetFiles, params):
        logger(f'Processing data using {modelName} model ...')

        # Reading data size from the selected dataset
        dataDictionary = readDataSizes(params['datasetName'], datasetFiles)
        users, pois = dataDictionary['users'], dataDictionary['pois']

        # Loading trainin items
        logger('Reading dataset instances ...')
        sparseTrainingMatrix, trainingMatrix = readSparseTrainingData(
            datasetFiles['train'], users['count'], pois['count'])
        trainingCheckins = readTrainingCheckins(
            datasetFiles['checkins'], sparseTrainingMatrix)
        sortedTrainingCheckins = {uid: sorted(trainingCheckins[uid], key=lambda k: k[1])
                                  for uid in trainingCheckins}
        socialRelations = readFriendData(
            datasetFiles['socialRelations'], 'list', None)
        groundTruth = readTestData(datasetFiles['test'])
        poiCoos = readPoiCoos(datasetFiles['poiCoos'])

        # Limit the number of users
        if (limitUsers != -1):
            logger(f'Limiting the number of users to {limitUsers} ...')
            users['count'] = limitUsers
            users['list'] = users['list'][:limitUsers]

        # Computing the final scores
        FCFScores = friendBasedCalculations(
            params['datasetName'], users, pois, socialRelations, poiCoos, sparseTrainingMatrix, groundTruth)
        KDEScores = kernelDensityEstimationCalculations(
            params['datasetName'], users, pois, poiCoos, sparseTrainingMatrix, groundTruth)
        AMCScores = additiveMarkovChainCalculations(
            params['datasetName'], users, pois, sortedTrainingCheckins, groundTruth)

        # Segmenting active users
        calculateActiveUsers(params['datasetName'], datasetFiles['train'])

        # Evaluation
        evalParams = {'usersList': users['list'], 'usersCount': users['count'],
                      'groundTruth': groundTruth, 'fusion': params['fusion'], 'poiList': pois['list'],
                      'trainingMatrix': trainingMatrix, 'evaluation': params['evaluation']}
        modelParams = {'FCF': FCFScores, 'KDE': KDEScores, 'AMC': AMCScores}
        evaluator(modelName, params['datasetName'], evalParams, modelParams)
