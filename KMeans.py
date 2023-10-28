import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class KMeans(object):
    def __init__(self, k):
        self.k = k
        self.idx = None
        self.center = None

    def fit(self, X, initial_center_index=None, epochs=50, seed=22):
        m, n = X.shape

        if initial_center_index is None:
            np.random.seed(seed)
            initial_center_index = np.random.randint(0, m, self.k)

        self.center = X[initial_center_index, :]

        plt.ion()
        for _ in tqdm(range(epochs)):
            self.idx = self.find_closest_center(X, self.center)
            self.compute_center(X)
        plt.show()
        return self.idx

    # noinspection PyMethodMayBeStatic
    def find_closest_center(self, X, center):
        # 这种方式利用 numpy 的广播机制，直接计算样本到各中心的距离，不用循环，速度比较快，但是在样本比较大时，更消耗内存
        distance = np.sum((X[:, np.newaxis, :] - center) ** 2, axis=2)
        idx = distance.argmin(axis=1)
        return idx

    def compute_center(self, X):
        self.center = np.zeros((self.k, X.shape[1]))
        for i in range(self.k):
            self.center[i, :] = np.mean(X[self.idx == i], axis=0)
