import numpy as np
from utils import logger
from config import limitUsers


def readDataSizes(datasetName: str, datasetFiles: dict):
    """
    Reads the data from the dataset files.

    Parameters
    ----------
    datasetName : str
        The name of the dataset.
    datasetFiles : dict
        The dataset files.

    Returns
    -------
        data : dict
    """
    print("Reading the 'dataSize' file to prepare further processing...")
    categoriesCount = 0
    # Loading required data based on the dataset name
    if (datasetName == 'Gowalla'):
        usersCount, poisCount = open(datasetFiles['dataSize'], 'r').readlines()[
            0].strip('\n').split()
    elif (datasetName == 'Yelp'):
        usersCount, poisCount, categoriesCount = open(
            datasetFiles['dataSize'], 'r').readlines()[0].strip('\n').split()
    # Converting data into integer
    usersCount, poisCount, categoriesCount = int(
        usersCount), int(poisCount), int(categoriesCount)
    # Creating lists for the data
    usersList, poisList, categoriesList = list(
        range(usersCount)), list(range(poisCount)), list(range(categoriesCount))
    # IMPORTANT: Shuffle the users list only if not limit is set
    if (limitUsers == -1):
        np.random.shuffle(usersList)
    # Providing feedback to the user
    logger(f'{datasetName} dataset contains {usersCount} users, {poisCount} locations, and {categoriesCount} categories!')
    # Returning the data
    return {'users': {'count': usersCount, 'list': usersList},
            'pois': {'count': poisCount, 'list': poisList},
            'categories': {'count': categoriesCount, 'list': categoriesList}
            }
