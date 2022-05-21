import time
import numpy as np
from utils import logger
from collections import defaultdict


class FriendBasedCF(object):
    def __init__(self, eta=0.5):
        self.eta = eta
        self.socialProximity = defaultdict(list)
        self.checkinMatrix = None

    def friendsSimilarityCalculation(self, socialRelations, checkinMatrix):
        startTime = time.time()
        logger('Calculating friends similarity ...')
        self.checkinMatrix = checkinMatrix
        for uid in socialRelations:
            for fid in socialRelations[uid]:
                if uid < fid:
                    u_socialNeighbors = set(socialRelations[uid])
                    f_socialNeighbors = set(socialRelations[fid])
                    jaccardFriend = (1.0 * len(u_socialNeighbors.intersection(f_socialNeighbors)) /
                                     len(u_socialNeighbors.union(f_socialNeighbors)))
                    u_checkinNeighbors = set(
                        checkinMatrix[uid, :].nonzero()[0])
                    f_checkinNeighbors = set(
                        checkinMatrix[fid, :].nonzero()[0])
                    jaccardCheckin = (1.0 * len(u_checkinNeighbors.intersection(f_checkinNeighbors)) /
                                      len(u_checkinNeighbors.union(f_checkinNeighbors)))
                    if jaccardFriend > 0 and jaccardCheckin > 0:
                        self.socialProximity[uid].append(
                            [fid, jaccardFriend, jaccardCheckin])
        elapsedTime = '{:.2f}'.format(time.time() - startTime)
        logger(f'Finished in {elapsedTime} seconds.')

    def predict(self, i, j):
        if i in self.socialProximity:
            numerator = np.sum([(self.eta * jf + (1 - self.eta) * jc) * self.checkinMatrix[k, j]
                                for k, jf, jc in self.socialProximity[i]])
            return numerator
        return 0.0
