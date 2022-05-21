import numpy as np
from utils import logger
from config import limitUsers
from Evaluations.evaluator import evaluator
from Data.readDataSizes import readDataSizes
from Models.USG.powerLaw import powerLawCalculations
from Models.USG.userBased import userBasedCalculations
from Models.USG.friendBased import friendBasedCalculations
from Data.calculateActiveUsers import calculateActiveUsers
from Models.utils import readTrainingData, readFriendData, readTestData, readPoiCoos

modelName = 'USG'


class USGMain:
    def main(datasetFiles, params):
        logger(f'Processing data using {modelName} model ...')

        # Reading data size from the selected dataset
        dataDictionary = readDataSizes(params['datasetName'], datasetFiles)
        users, pois = dataDictionary['users'], dataDictionary['pois']

        # Loading training items
        logger('Reading dataset instances ...')
        trainingMatrix = readTrainingData(
            datasetFiles['train'], users['count'], pois['count'], False)
        groundTruth = readTestData(datasetFiles['test'])
        socialRelations = readFriendData(
            datasetFiles['socialRelations'], 'dictionary', None)
        poiCoos = readPoiCoos(datasetFiles['poiCoos'])

        # Limit the number of users
        if (limitUsers != -1):
            logger(f'Limiting the number of users to {limitUsers} ...')
            users['count'] = limitUsers
            users['list'] = users['list'][:limitUsers]

        # Computing the final scores
        UScores = userBasedCalculations(
            params['datasetName'], users, pois, trainingMatrix, groundTruth)
        SScores = friendBasedCalculations(
            params['datasetName'], users, pois, socialRelations, trainingMatrix, groundTruth)
        GScores = powerLawCalculations(
            params['datasetName'], users, pois, trainingMatrix, poiCoos, groundTruth)

        # Segmenting active users
        calculateActiveUsers(params['datasetName'], datasetFiles['train'])

        # Evaluation
        evalParams = {'usersList': users['list'], 'usersCount': users['count'],
                      'groundTruth': groundTruth, 'fusion': params['fusion'], 'poiList': pois['list'],
                      'trainingMatrix': trainingMatrix, 'evaluation': params['evaluation']}
        modelParams = {'U': UScores, 'S': SScores, 'G': GScores}
        evaluator(modelName, params['datasetName'], evalParams, modelParams)
