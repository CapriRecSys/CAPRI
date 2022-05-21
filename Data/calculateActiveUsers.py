import os
import numpy as np
import pandas as pd
from utils import logger
from config import activeUsersPercentage


def calculateActiveUsers(dataset: str, trainFilePath: str):
    """
    Calculates the number of active users in the dataset.

    Parameters
    ----------
    dataset : str
        The name of the dataset.
    trainFilePath : str
        The path to the training data file.
    """
    # Check if activeUsersPercentage is set
    if (len(activeUsersPercentage) < 1):
        return
    # Create a log
    logger(f'Calculating active/inactive users for dataset {dataset} ...')
    # Read train data and set to DataFrame
    dataFrame = pd.read_csv(trainFilePath, sep="\t", names=[
                            'userId', 'locId', 'freq'])
    dataFrame = dataFrame.drop(columns=['locId'])
    # Group dataFrame based on userId
    processedDataFrame = dataFrame.groupby(['userId'], as_index=False).sum()
    # Sort dataFrame based on frequency
    processedDataFrame = processedDataFrame.sort_values(
        by=['freq'], ascending=False)
    # Obtain the ordered list of userIds
    orderedList = processedDataFrame['userId'].tolist()
    # Iterate over activeUsersPercentage
    for percentage in activeUsersPercentage:
        # Check if has not been previously calculated
        normalPercentage = '{0:02d}'.format(percentage)
        fileName = f'{dataset}_Active_{normalPercentage}Percent.txt'
        path = os.path.abspath(f'./Data/_processedData/{fileName}')
        if (os.path.exists(path)):
            logger(f'{fileName} already calculated!')
            continue
        # Calculate active users
        limit = int(processedDataFrame.shape[0] * percentage / 100)
        activeUsers = orderedList[:limit]
        inactiveUsers = orderedList[limit:]
        # Sort main dataFrame based on orderedList
        # NOTE: We need to keep the order of the users based on the orderedList, so we should set userId as index
        print(f'Calculating {fileName} ...')
        activeUsersDF = dataFrame.set_index(
            'userId').loc[activeUsers].reset_index()
        inactiveUsersDF = dataFrame.set_index(
            'userId').loc[inactiveUsers].reset_index()
        # Limit users based on limit and export to file
        print('Exporting to file ...')
        activeUsersDF.to_csv(
            path, sep='\t', index=False, header=False)
        inactiveUsersDF.to_csv(path.replace(
            'Active', 'Inactive'), sep='\t', index=False, header=False)
