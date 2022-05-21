import time
import math
import numpy as np
from utils import logger
from collections import defaultdict


def dist(loc1, loc2):
    lat1, long1 = loc1
    lat2, long2 = loc2
    if abs(lat1 - lat2) < 1e-6 and abs(long1 - long2) < 1e-6:
        return 0.0
    degreesToRadians = math.pi/180.0
    phi1 = (90.0 - lat1) * degreesToRadians
    phi2 = (90.0 - lat2) * degreesToRadians
    theta1 = long1 * degreesToRadians
    theta2 = long2 * degreesToRadians
    cos = (math.sin(phi1) * math.sin(phi2) * math.cos(theta1 - theta2) +
           math.cos(phi1) * math.cos(phi2))
    arc = math.acos(cos)
    earthRadius = 6371
    return arc * earthRadius


class FriendBasedCF(object):
    def __init__(self):
        self.socialProximity = defaultdict(list)
        self.sparseCheckinMatrix = None

    def friendsSimilarityCalculation(self, socialRelations, poiCoos, sparseCheckinMatrix):
        self.sparseCheckinMatrix = sparseCheckinMatrix
        startTime = time.time()
        logger('Calculating friends similarity ...')
        residenceLids = np.asarray(
            sparseCheckinMatrix.tocsr().argmax(axis=1)).reshape(-1)
        residenceCoos = [poiCoos[lid] for lid in residenceLids.tolist()]
        maxDistance = [-1.0 for _ in range(sparseCheckinMatrix.shape[0])]

        for uid1, uid2 in socialRelations:
            dis = dist(residenceCoos[uid1], residenceCoos[uid2])
            maxDistance[uid1] = max(maxDistance[uid1], dis)
            maxDistance[uid2] = max(maxDistance[uid2], dis)
            self.socialProximity[uid1].append([uid2, dis])
            self.socialProximity[uid2].append([uid1, dis])

        for uid in self.socialProximity:
            # Max distance + 1 to smooth.
            self.socialProximity[uid] = [[fid, 1.0 - (dis / (1.0 + maxDistance[uid]))]
                                         for fid, dis in self.socialProximity[uid]]
        elapsedTime = '{:.2f}'.format(time.time() - startTime)
        logger(f'Finished in {elapsedTime} seconds.')

    def predict(self, i, j):
        if i in self.socialProximity:
            numerator = np.sum([weight * self.sparseCheckinMatrix[k, j]
                               for k, weight in self.socialProximity[i]])
            denominator = np.sum(
                [weight for k, weight in self.socialProximity[i]])
            return numerator / denominator
        return 0.0
