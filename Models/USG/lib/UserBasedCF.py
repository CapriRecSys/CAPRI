import time
from utils import logger
from numpy.linalg import norm


class UserBasedCF(object):
    def __init__(self):
        self.recScore = None

    def loadModel(self, loadedModel):
        self.recScore = loadedModel

    def preComputeRecScores(self, C):
        startTime = time.time()
        logger('Training User-based Collaborative Filtering ...')
        sim = C.dot(C.T)
        norms = [norm(C[i]) for i in range(C.shape[0])]
        for i in range(C.shape[0]):
            sim[i][i] = 0.0
            for j in range(i+1, C.shape[0]):
                sim[i][j] /= (norms[i] * norms[j])
                sim[j][i] /= (norms[i] * norms[j])
        self.recScore = sim.dot(C)
        elapsedTime = '{:.2f}'.format(time.time() - startTime)
        logger(f'Finished in {elapsedTime} seconds.')

    def predict(self, i, j):
        return self.recScore[i][j]
