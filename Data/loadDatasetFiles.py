import os
from utils import logger
from config import datasets, dataDirectory


def loadDatasetFiles(datasetName: str):
    """
    Loads the selected dataset files from their corresponding data directory.

    Parameters
    ----------
    datasetName : str
        The name of the dataset to be loaded.

    Returns
    -------
    dataset : dict
        The loaded dataset.
    """
    datasetFiles = {}
    # Check if the dataset is among the available ones
    supportedDatasets = list(datasets.keys())
    if not datasetName in supportedDatasets:
        logger(f'{datasetName} was not among the supported datasets!', 'error')
        return
    # Otherwise, listing the files existing in dataset directories
    print(f'Loading {datasetName} dataset files ...')
    fileList = os.listdir(f'{dataDirectory}/{datasetName}')
    # Creating a dictionary of dataset items
    for file in fileList:
        fileName = os.path.basename(file).split('.')[0]
        # Adding the absolute path to the items: e.g. 'checkins.txt' ==> 'C\\...\\Datacheckins.txt'
        datasetFiles[fileName] = f'{dataDirectory}\\{datasetName}\\{file}'
    print(f'{datasetName} dataset files have been loaded for processing!')
    return datasetFiles
