import numpy as np
from numpy import linalg

class PCA:

    def __init__(self, dataset):
        self.dataset = np.matrix(dataset, dtype='float64').T

    def principal_comps(self, threshold = 0.85):
        ret = []
        data = []
        for (index, line) in enumerate(self.dataset):
            self.dataset[index] -= np.mean(line)
            self.dataset[index] /= np.std(line, ddof = 1)
        Cov = np.cov(self.dataset)
        eigs, vectors = linalg.eig(Cov)
        for i in range(len(eigs)):
            data.append((eigs[i], vectors[:, i].T))
        data.sort(key = lambda x: x[0], reverse = True)
        sum = 0
        for comp in data:
            sum += comp[0] / np.sum(eigs)
            ret.append(
                tuple(map(
                    lambda x: np.round(x, 5),
                    (comp[1], comp[0] / np.sum(eigs), sum)
                ))
            )
            print('特征值:', comp[0], '特征向量:', ret[-1][0], '方差贡献率:', ret[-1][1], '累计方差贡献率:', ret[-1][2])
            if sum > threshold:
                return ret
        return ret
