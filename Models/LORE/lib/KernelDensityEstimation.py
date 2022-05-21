import time
import math
import numpy as np
from utils import logger
from collections import defaultdict


class KernelDensityEstimation(object):
    def __init__(self):
        self.poiCoos = None
        self.L = None
        self.bw = None

    def precomputeKernelParameters(self, sparseCheckinMatrix, poiCoos):
        self.poiCoos = poiCoos
        startTime = time.time()
        logger('Pre-computing kernel parameters ...')
        trainingLocations = defaultdict(list)
        for uid in range(sparseCheckinMatrix.shape[0]):
            trainingLocations[uid] = [poiCoos[lid]
                                      for lid in sparseCheckinMatrix[uid].nonzero()[1].tolist()]
        L = trainingLocations
        bw = {}
        for u in L:
            if len(L[u]) > 1:
                std = np.std([coo for coo in L[u]], axis=0)
                bw[u] = 1.0 / (len(L[u])**(1.0/6)) * \
                    np.sqrt(0.5 * std.dot(std))
        elapsedTime = '{:.2f}'.format(time.time() - startTime)
        logger(f'Finished in {elapsedTime} seconds.')
        self.L = L
        self.bw = bw

    def K(self, x):
        return np.exp(-0.5 * np.sum(x * x, axis=1)) / (2 * math.pi)

    def predict(self, u, lj):
        if u in self.L and u in self.bw:
            lat_j, lng_j = self.poiCoos[lj]
            x = [np.array([lat_i - lat_j, lng_i - lng_j]) / self.bw[u]
                 for lat_i, lng_i in self.L[u]]
            return sum(self.K(np.array(x))) / len(self.L[u]) / (self.bw[u] ** 2)
        return 1.0
