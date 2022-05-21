import time
import numpy as np
from utils import logger
from collections import defaultdict


class AdditiveMarkovChain(object):
    def __init__(self, deltaT, alpha):
        self.S = None
        self.alpha = alpha
        self.deltaT = deltaT
        self.OCount, self.TCount = None, None

    def buildLocationToLocationTransitionGraph(self, sortedTrainingCheckins):
        startTime = time.time()
        logger('Building the location-location transition graph (L2TG) ...')
        S = sortedTrainingCheckins
        OCount = defaultdict(int)
        TCount = defaultdict(lambda: defaultdict(int))
        for u in S:
            lastL, lastT = S[u][0]
            for i in range(1, len(S[u])):
                l, t = S[u][i]
                if t - lastT <= self.deltaT:
                    OCount[lastL] += 1
                    TCount[lastL][l] += 1
                lastL, lastT = l, t
        elapsedTime = '{:.2f}'.format(time.time() - startTime)
        logger(f'Finished in {elapsedTime} seconds.')
        self.S = S
        self.OCount = OCount
        self.TCount = TCount

    def TP(self, l, nextL):
        if l not in self.OCount:
            return 1.0 if l == nextL else 0.0
        elif l in self.TCount and nextL in self.TCount[l]:
            return 1.0 * self.TCount[l][nextL] / self.OCount[l]
        else:
            return 0.0

    def W(self, i, n):
        return np.exp2(-self.alpha * (n - i))

    def predict(self, u, l):
        if u in self.S:
            n = len(self.S[u])
            numerator = np.sum([self.W(i, n) * self.TP(li, l)
                               for i, (li, _) in enumerate(self.S[u])])
            denominator = np.sum([self.W(i, n) for i in range(len(self.S[u]))])
            return 1.0 * numerator / denominator
        return 1.0
