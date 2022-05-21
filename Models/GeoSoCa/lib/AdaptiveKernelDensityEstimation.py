import time
import math
import numpy as np
from utils import logger
from collections import defaultdict


class AdaptiveKernelDensityEstimation(object):
    def __init__(self, alpha=0.5):
        self.R = None
        self.N = None
        self.h = None
        self.alpha = alpha
        self.poiCoos = None
        self.checkinMatrix = None
        self.H1, self.H2 = None, None

    def precomputeKernelParameters(self, checkinMatrix, poiCoos):
        self.poiCoos = poiCoos
        startTime = time.time()
        logger('Preparing Adaptive Kernel Density Estimation ...')
        trainingLocations = defaultdict(list)
        for uid in range(checkinMatrix.shape[0]):
            trainingLocations[uid] = [[lid, np.array(poiCoos[lid])]
                                      for lid in checkinMatrix[uid].nonzero()[0].tolist()]
        N = {uid: np.sum(checkinMatrix[uid])
             for uid in range(checkinMatrix.shape[0])}
        self.checkinMatrix = checkinMatrix
        self.N = N
        R = trainingLocations
        self.R = R
        H1, H2 = {}, {}
        for uid in R:
            meanCoo = np.sum([checkinMatrix[uid, lid] * coo
                              for lid, coo in R[uid]], axis=0) / N[uid]
            meanCooSqDiff = np.sum([checkinMatrix[uid, lid] * (coo - meanCoo) ** 2
                                    for lid, coo in R[uid]], axis=0) / N[uid]
            H1[uid], H2[uid] = 1.06 / \
                (len(R[uid])**0.2) * np.sqrt(meanCooSqDiff)
        self.H1, self.H2 = H1, H2
        h = defaultdict(lambda: defaultdict(int))
        for uid in R:
            if not H1[uid] == 0 and not H2[uid] == 0:
                fGeoValues = {li[0]: self.fGeoWithFixedBandwidth(
                    uid, li, R) for li in R[uid]}
                g = np.prod(list(fGeoValues.values())) ** (1.0 / len(R[uid]))
                for lid, f_geo_val in fGeoValues.items():
                    h[uid][lid] = (g / f_geo_val) ** self.alpha
        self.h = h
        elapsedTime = '{:.2f}'.format(time.time() - startTime)
        logger(f'Finished in {elapsedTime} seconds.')

    def fGeoWithFixedBandwidth(self, u, l, R):
        l, (lat, lng) = l
        return np.sum([self.checkinMatrix[u, li] * self.K_H(u, lat, lng, lat_i, lng_i)
                       for li, (lat_i, lng_i) in R[u]]) / self.N[u]

    def fGeoWithLocalBandwidth(self, u, l, R):
        l, (lat, lng) = l
        return np.sum([self.checkinMatrix[u, li] * self.K_Hh(u, lat, lng, lat_i, lng_i, li)
                       for li, (lat_i, lng_i) in R[u]]) / self.N[u]

    def K_H(self, u, lat, lng, lat_i, lng_i):
        return (1.0 / (2 * math.pi * self.H1[u] * self.H2[u]) *
                np.exp(-((lat - lat_i)**2 / (2 * self.H1[u]**2)) -
                       ((lng - lng_i)**2 / (2 * self.H2[u]**2))))

    def K_Hh(self, u, lat, lng, lat_i, lng_i, li):
        return (1.0 / (2 * math.pi * self.H1[u] * self.H2[u] * self.h[u][li]**2) *
                np.exp(-((lat - lat_i)**2 / (2 * self.H1[u]**2 * self.h[u][li]**2)) -
                       ((lng - lng_i)**2 / (2 * self.H2[u]**2 * self.h[u][li]**2))))

    def predict(self, u, l):
        if not self.H1[u] == 0 and not self.H2[u] == 0 and not sum(self.h[u].values()) == 0:
            l = [l, self.poiCoos[l]]
            return self.fGeoWithLocalBandwidth(u, l, self.R)
        return 1.0
