import logging
from utils import logger
from commandParser import getUserChoices
from Data.loadDatasetFiles import loadDatasetFiles


def __init__():
    # Creating log file
    logging.basicConfig(filename='logger.log', level=logging.INFO)
    logger('CAPRI framework started!')
    # Fetching user choices
    userInputs = getUserChoices()
    # Checking if user wants to load a model
    if (userInputs != None):
        # Initializing dataset items
        datasetFiles = loadDatasetFiles(userInputs['Dataset'])
        logger(f'Dataset files: {datasetFiles}', 'info', True)
        # Exiting the program if dataset is not found
        if (datasetFiles == None):
            return
        # Initializing parameters
        parameters = {
            "fusion": userInputs['Fusion'],
            "ignored": userInputs['Ignored'],
            "datasetName": userInputs['Dataset'],
            "evaluation": userInputs['Evaluation'],
        }
        logger(f'Processing parameters: {parameters}', 'info', True)
        # Dynamically loading the model
        module = __import__(
            'Models.' + userInputs['Model'] + '.main', fromlist=[''])
        selectedModule = getattr(module, userInputs['Model'] + 'Main')
        # Select the 'main' class in the module
        selectedModule.main(datasetFiles, parameters)
        # Closing the log file
        logger('CAPRI framework finished!')
    else:
        logger('Framework stopepd!')


__init__()
