import numpy as np
import matplotlib.pyplot as plt


class PCA(object):
    def __init__(self, n_components):
        self.n_components = n_components
        self.x_mean = None
        self.x_std = None
        self.vals = None
        self.vecs = None

    def fit(self, X):
        self.x_mean = np.mean(X, axis=0)
        self.x_std = np.std(X, axis=0)

        X = (X - self.x_mean) / (self.x_std + 1e-5)
        m = X.shape[0]
        sigma = np.dot(X.T, X) / m  # 求协方差矩阵
        self.vals, self.vecs = np.linalg.eig(sigma)  # 特征值，特征向量

        return self

    def transform(self, X):
        # 数据降维操作
        return np.dot(X, self.vecs[:, :self.n_components])

    def recover(self, Z):
        # 将降维的数据，通过映射矩阵重新映射回原来的维度，方便可视化对比降维前后的效果
        return np.dot(Z, self.vecs[:, :self.n_components].T)


def display_data(X, title=None):
    m, n = X.shape
    size = int(np.sqrt(n))
    display_rows = 10
    display_cols = int(m / 10)
    display = np.zeros((size * display_rows, size * display_cols))

    temp = 0
    for i in range(display_rows):
        for j in range(display_cols):
            display[i * size: (i + 1) * size, j * size: (j + 1) * size] = X[temp, :].reshape(size, -1).T
            temp += 1

    plt.title(title)
    plt.imshow(display)
    plt.show()
