import numpy as np
from utils import logger
from config import limitUsers
from Evaluations.evaluator import evaluator
from Data.readDataSizes import readDataSizes
from Models.GeoSoCa.social import socialCalculations
from Data.calculateActiveUsers import calculateActiveUsers
from Models.GeoSoCa.categorical import categoricalCalculations
from Models.GeoSoCa.geographical import geographicalCalculations
from Models.utils import readPoiCoos, readTestData, readCategoryData, readTrainingData, readFriendData

modelName = 'GeoSoCa'


class GeoSoCaMain:
    def main(datasetFiles, params):
        logger(f'Processing data using {modelName} model ...')

        # Reading data size from the selected dataset
        dataDictionary = readDataSizes(params['datasetName'], datasetFiles)
        users, pois, categories = dataDictionary['users'], dataDictionary['pois'], dataDictionary['categories']

        # Skipped context
        skipCategory = bool(categories['count'] == 0)

        # Loading data from the selected dataset
        logger('Reading dataset instances ...')
        poiCoos = readPoiCoos(datasetFiles['poiCoos'])
        trainingMatrix = readTrainingData(
            datasetFiles['train'], users['count'], pois['count'], True)
        socialRelations = readFriendData(
            datasetFiles['socialRelations'], 'ndarray', users['count'])
        groundTruth = readTestData(datasetFiles['test'])
        # If the dataset does not cover categories, do not read them
        poiCategoryMatrix = np.empty((0, 0))
        if not skipCategory:
            poiCategoryMatrix = readCategoryData(
                datasetFiles['poiCategories'], categories['count'], pois['count'])

        # Limit the number of users
        if (limitUsers != -1):
            logger(f'Limiting the number of users to {limitUsers} ...')
            users['count'] = limitUsers
            users['list'] = users['list'][:limitUsers]

        # Computing the final scores
        AKDEScores = geographicalCalculations(
            params['datasetName'], users, pois, poiCoos, trainingMatrix, groundTruth)
        SCScores = socialCalculations(
            params['datasetName'], users, pois, trainingMatrix, socialRelations, groundTruth)
        CCScores = None
        if not skipCategory:
            CCScores = categoricalCalculations(
                params['datasetName'], users, pois, trainingMatrix, poiCategoryMatrix, groundTruth)

        # Segmenting active users
        calculateActiveUsers(params['datasetName'], datasetFiles['train'])

        # Evaluation
        evalParams = {'usersList': users['list'], 'usersCount': users['count'],
                      'groundTruth': groundTruth, 'fusion': params['fusion'], 'poiList': pois['list'],
                      'trainingMatrix': trainingMatrix, 'evaluation': params['evaluation']}
        modelParams = {'AKDE': AKDEScores, 'SC': SCScores, 'CC': CCScores}
        evaluator(modelName, params['datasetName'], evalParams, modelParams)
