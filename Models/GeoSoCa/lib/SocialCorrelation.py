import time
import numpy as np
from utils import logger


class SocialCorrelation(object):
    def __init__(self):
        self.beta = None
        self.X = None

    def loadModel(self, loadedModel):
        self.X = loadedModel
        self.beta = 1.0 + 1.0 / np.mean(np.log(1.0 + self.X[self.X > 0]))

    def computeBeta(self, checkinMatrix, socialMatrix):
        startTime = time.time()
        logger('Preparing Social Correlation Parameter ...')
        S = socialMatrix
        R = checkinMatrix
        X = S.dot(R)
        beta = 1.0 + 1.0 / np.mean(np.log(1.0 + X[X > 0]))
        self.beta = beta
        self.X = X
        elapsedTime = '{:.2f}'.format(time.time() - startTime)
        logger(f'Finished in {elapsedTime} seconds.')

    def predict(self, u, l):
        return 1.0 - (1.0 + self.X[u, l]) ** (1 - self.beta)
